"""
==========================================================
 Deep Blue Synthetix — Customer Support AI Orchestrator
==========================================================
 FastAPI backend that orchestrates:
   1. Multi-Task DistilBERT classification (Category + Urgency)
   2. ChromaDB semantic retrieval (Knowledge Base citations)
   3. Groq LLM generation (Summary + Grounded Draft Reply)

 Run:
     python main.py
 Docs:
     http://localhost:8000/docs   (Swagger UI)
     http://localhost:8000/redoc  (ReDoc)
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional, List

import uvicorn
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizerFast
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the model architecture from the training script
from train_classifier import MultiHeadDistilBERT

# ──────────────────────────────────────────────
#  Configuration & Logging
# ──────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepBlueSynthetix")

# ──────────────────────────────────────────────
#  Global State (loaded once at startup)
# ──────────────────────────────────────────────
state = {
    "model": None,
    "tokenizer": None,
    "label_maps": None,
    "embedding_model": None,
    "collection": None,
    "device": None,
    "models_loaded": False,  # Flag for fallback logic
}

# ──────────────────────────────────────────────
#  Pydantic Schemas
# ──────────────────────────────────────────────

class TicketRequest(BaseModel):
    """Schema for an incoming support ticket."""
    subject: str = Field(
        default="",
        description="The subject line of the support ticket.",
        json_schema_extra={"example": "Cannot reset my password"}
    )
    description: str = Field(
        ...,
        description="The full body/description of the support ticket.",
        json_schema_extra={"example": "I have been trying to reset my password for 2 days. The reset link never arrives in my email."}
    )
    channel: Optional[str] = Field(
        default=None,
        description="The channel the ticket was submitted through (e.g., Email, Chat, Social Media).",
        json_schema_extra={"example": "Email"}
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of when the ticket was created.",
        json_schema_extra={"example": "2026-03-05T18:30:00+05:30"}
    )


class TicketResponse(BaseModel):
    """Schema for the AI-processed ticket response."""
    category: str = Field(..., description="Predicted ticket category (e.g., Incident, Request, Problem, Change).")
    urgency: str = Field(..., description="Predicted urgency level (High, Medium, Low).")
    summary: str = Field(..., description="AI-generated 1-sentence summary of the ticket.")
    draft_reply: str = Field(..., description="AI-drafted reply grounded in the Knowledge Base.")
    sources: List[str] = Field(..., description="List of KB document filenames used as citations.")
    confidence: float = Field(..., description="Softmax confidence percentage for the category prediction.")


# ──────────────────────────────────────────────
#  FastAPI Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="Deep Blue Synthetix API",
    description=(
        "AI-Powered Customer Support Ticket Orchestrator & Auto-Responder.\n\n"
        "This API classifies incoming support tickets by **Category** and **Urgency** "
        "using a fine-tuned Multi-Task DistilBERT model, retrieves relevant Knowledge Base "
        "articles via ChromaDB semantic search, and drafts a grounded reply using **Llama 3** (via Groq)."
    ),
    version="2.0.0",
    contact={"name": "Deep Blue Synthetix Team"},
)

# ──────────────────────────────────────────────
#  CORS Middleware (Frontend Access)
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins for hackathon demo
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

# ──────────────────────────────────────────────
#  Startup: Load All Models Into Memory
# ──────────────────────────────────────────────

@app.on_event("startup")
def setup_models():
    """Load the DistilBERT classifier, SentenceTransformer, and ChromaDB at server boot."""
    logger.info("🚀 Initializing models and database...")
    try:
        # Determine compute device
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        state["device"] = device
        logger.info(f"🖥  Using device: {device}")

        # ── 1. Classification Model ────────────────
        model_dir = "./models/ticket_classifier"
        model_path = os.path.join(model_dir, "model.pt")
        label_maps_path = os.path.join(model_dir, "label_maps.json")

        model_obj = MultiHeadDistilBERT()
        model_obj.load_state_dict(torch.load(model_path, map_location=device))
        model_obj.to(device)
        model_obj.eval()
        state["model"] = model_obj

        state["tokenizer"] = DistilBertTokenizerFast.from_pretrained(model_dir)

        with open(label_maps_path, "r") as f:
            state["label_maps"] = json.load(f)

        logger.info("✅ Classification model loaded successfully.")

        # ── 2. Embedding Model & Vector DB ─────────
        state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
        chroma_client = chromadb.PersistentClient(path="./models/vector_db")
        state["collection"] = chroma_client.get_collection(name="support_kb")

        logger.info("✅ SentenceTransformer & ChromaDB loaded successfully.")

        state["models_loaded"] = True
        logger.info("🟢 All systems operational. Server is ready.")

    except Exception as e:
        logger.error(f"❌ Startup failure: {e}")
        logger.warning("⚠️  Server will start with MOCK fallback responses.")
        state["models_loaded"] = False


# ──────────────────────────────────────────────
#  Helper: Predict Ticket (DistilBERT)
# ──────────────────────────────────────────────

def predict_ticket(text: str) -> dict:
    """Run the Multi-Task DistilBERT classifier on the input text."""
    device = state["device"]
    model = state["model"]
    tokenizer = state["tokenizer"]
    label_maps = state["label_maps"]

    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        cat_logits, urg_logits = model(input_ids, attention_mask)

        # Softmax for confidence scoring
        cat_probs = F.softmax(cat_logits, dim=1)
        confidence = torch.max(cat_probs).item() * 100.0

        cat_id = cat_logits.argmax(dim=1).item()
        urg_id = urg_logits.argmax(dim=1).item()

    category = label_maps["id2cat"][str(cat_id)]
    urgency = label_maps["id2urg"][str(urg_id)]

    return {"category": category, "urgency": urgency, "confidence": round(confidence, 2)}


# ──────────────────────────────────────────────
#  Helper: Retrieve Context (ChromaDB)
# ──────────────────────────────────────────────

def retrieve_context(text: str, top_k: int = 2) -> dict:
    """Embed the query and retrieve the top-K KB snippets from ChromaDB."""
    embedding_model = state["embedding_model"]
    collection = state["collection"]

    query_embedding = embedding_model.encode([text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    sources = list({m.get("source", "Unknown") for m in metadatas})

    return {"documents": documents, "sources": sources}


# ──────────────────────────────────────────────
#  Helper: Generate LLM Reply (Groq / Llama 3)
# ──────────────────────────────────────────────

def generate_llm_reply(text: str, context: list) -> dict:
    """Use the Groq LLM to generate a grounded draft reply and a 1-sentence summary."""
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key or api_key == "your_api_key_here":
        return {
            "draft_reply": "⚠️ GROQ_API_KEY is not configured. Set it in your .env file.",
            "summary": "⚠️ GROQ_API_KEY is not configured.",
        }

    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        context_str = "\n\n---\n\n".join(context) if context else "No context found."

        # ── Draft Reply ────────────────────────
        reply_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional, friendly customer support agent for Deep Blue Synthetix. "
             "Using the provided Knowledge Base context below, draft a helpful and empathetic reply to the customer's issue. "
             "Include specific steps, links, or policy details from the context when relevant. "
             "Always sign off as 'Deep Blue Synthetix Support Team' — never use placeholders like '[Your Name]'. "
             "If the context does not contain ANY information related to the customer's issue at all, "
             "respond EXACTLY with: \"I need more clarification to resolve this.\" "
             "Do NOT invent information that is not in the context.\n\nKnowledge Base Context:\n{context}"),
            ("human", "User Issue: {issue}"),
        ])
        draft_reply = (reply_prompt | llm | StrOutputParser()).invoke(
            {"context": context_str, "issue": text}
        )

        # ── Summary ───────────────────────────
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Write a single short sentence summary of the user's issue."),
            ("human", "User Issue: {issue}"),
        ])
        summary = (summary_prompt | llm | StrOutputParser()).invoke({"issue": text})

        return {"draft_reply": draft_reply, "summary": summary}

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return {
            "draft_reply": f"Error generating reply: {str(e)}",
            "summary": f"Error generating summary: {str(e)}",
        }


# ──────────────────────────────────────────────
#  Mock Fallback (when models aren't loaded)
# ──────────────────────────────────────────────

def get_mock_response(text: str) -> TicketResponse:
    """Return a plausible mock response for demo/testing when ML models are unavailable."""
    logger.warning("Using MOCK fallback response.")
    return TicketResponse(
        category="Request",
        urgency="Medium",
        summary=f"Customer reported an issue: '{text[:50]}...'",
        draft_reply="Thank you for reaching out. A support agent will review your ticket shortly.",
        sources=["mock_fallback.txt"],
        confidence=0.0,
    )


# ══════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════

@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="Returns the current status of the API server and whether ML models are loaded.",
)
async def health_check():
    return {
        "status": "ok",
        "models_loaded": state["models_loaded"],
        "device": state.get("device", "N/A"),
        "timestamp": datetime.now().isoformat(),
    }


@app.post(
    "/process-ticket",
    response_model=TicketResponse,
    tags=["Ticket Processing"],
    summary="Process a Support Ticket",
    description=(
        "Accepts a support ticket and returns the AI-predicted **category**, **urgency**, "
        "a 1-sentence **summary**, a grounded **draft reply**, **source citations**, "
        "and a **confidence** percentage."
    ),
)
async def process_ticket(request: TicketRequest):
    """
    Main orchestration endpoint.
    Pipeline: Preprocess → Classify → Retrieve → Generate → Respond
    """
    logger.info(f"📩 New ticket received | Channel: {request.channel or 'N/A'}")

    # ── Combine inputs ─────────────────────
    combined_text = f"Subject: {request.subject}\n\nDescription: {request.description}"

    # ── Fallback if models didn't load ─────
    if not state["models_loaded"]:
        return get_mock_response(combined_text)

    try:
        # ── Step 1: Classification ─────────
        logger.info("  🧠 Running DistilBERT classifier...")
        prediction = predict_ticket(combined_text)
        logger.info(f"  ✅ Category: {prediction['category']} | Urgency: {prediction['urgency']} | Confidence: {prediction['confidence']}%")

        # ── Step 2: KB Retrieval ───────────
        logger.info("  📚 Querying ChromaDB for relevant KB articles...")
        retrieval = retrieve_context(combined_text)
        logger.info(f"  ✅ Retrieved {len(retrieval['documents'])} snippets from sources: {retrieval['sources']}")

        # ── Step 3: LLM Generation ─────────
        logger.info("  💬 Generating LLM summary and draft reply via Groq...")
        llm_output = generate_llm_reply(combined_text, retrieval["documents"])
        logger.info("  ✅ LLM generation complete.")

        # ── Step 4: Build Response ─────────
        return TicketResponse(
            category=prediction["category"],
            urgency=prediction["urgency"],
            summary=llm_output["summary"],
            draft_reply=llm_output["draft_reply"],
            sources=retrieval["sources"],
            confidence=prediction["confidence"],
        )

    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing ticket: {str(e)}")


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.critical(f"Failed to start server: {e}")
