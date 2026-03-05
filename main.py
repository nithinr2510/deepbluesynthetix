import json
import os
import uvicorn
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast
from sentence_transformers import SentenceTransformer
import chromadb

# Import the model architecture from the training script
from train_classifier import MultiHeadDistilBERT

# Global state to hold models
state = {
    "model": None,
    "tokenizer": None,
    "label_maps": None,
    "embedding_model": None,
    "collection": None,
    "device": None
}

app = FastAPI(
    title="Customer Support AI Orchestrator",
    description="Processes incoming support tickets using a Classifier and a Vector DB.",
    version="1.0.0"
)

@app.on_event("startup")
def setup_models():
    print("🚀 Initializing models and database...")
    try:
        # Determine device
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        state["device"] = device
        print(f"🖥  Using device: {device}")

        # 1. Load PyTorch Classification Model & Tokenizer
        model_dir = "./models/ticket_classifier"
        model_path = os.path.join(model_dir, "model.pt")
        label_maps_path = os.path.join(model_dir, "label_maps.json")
        
        # Initialize model architecture and load weights
        model_obj = MultiHeadDistilBERT()
        model_obj.load_state_dict(torch.load(model_path, map_location=device))
        model_obj.to(device)
        model_obj.eval()
        state["model"] = model_obj
        
        # Load tokenizer
        state["tokenizer"] = DistilBertTokenizerFast.from_pretrained(model_dir)
        
        # Load label maps
        with open(label_maps_path, "r") as f:
            state["label_maps"] = json.load(f)
            
        print("✅ Classification models loaded.")

        # 2. Load Embeddings & ChromaDB
        print("Initializing Local ChromaDB and SentenceTransformer...")
        state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
        
        db_dir = "./models/vector_db"
        chroma_client = chromadb.PersistentClient(path=db_dir)
        state["collection"] = chroma_client.get_collection(name="support_kb")
        
        print("✅ Context Retrieval models loaded.")
    except Exception as e:
        print(f"❌ Error during startup memory loading: {e}")
        raise e
class TicketRequest(BaseModel):
    ticket_text: str

class TicketResponse(BaseModel):
    category: str
    urgency: str
    retrieved_context: list[str]

@app.post("/process-ticket", response_model=TicketResponse)
async def process_ticket(request: TicketRequest):
    try:
        device = state["device"]
        model = state["model"]
        tokenizer = state["tokenizer"]
        label_maps = state["label_maps"]
        
        # ---------------------------------------------
        # A. Classification
        # ---------------------------------------------
        inputs = tokenizer(
            request.ticket_text,
            max_length=128,  # Using MAX_LEN from train_classifier.py
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            cat_logits, urg_logits = model(input_ids, attention_mask)
            cat_id = cat_logits.argmax(dim=1).item()
            urg_id = urg_logits.argmax(dim=1).item()
            
        # Extract string labels from ID
        # Note: JSON keys are strings, so we convert the integer id to string to index the dict
        category = label_maps["id2cat"][str(cat_id)]
        urgency = label_maps["id2urg"][str(urg_id)]

        # ---------------------------------------------
        # B. Retrieval
        # ---------------------------------------------
        embedding_model = state["embedding_model"]
        collection = state["collection"]
        
        query_embedding = embedding_model.encode([request.ticket_text]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=2 # Retrieve top 2 matches
        )
        
        retrieved_context = results['documents'][0] if results['documents'] else []

        # ---------------------------------------------
        # C. Return Response
        # ---------------------------------------------
        return TicketResponse(
            category=category,
            urgency=urgency,
            retrieved_context=retrieved_context
        )

    except Exception as e:
        # Proper error handling around inference and querying
        raise HTTPException(status_code=500, detail=f"Error processing ticket: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running properly."}

if __name__ == '__main__':
    # Wrap with lots of error handling
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"Failed to start server: {e}")
