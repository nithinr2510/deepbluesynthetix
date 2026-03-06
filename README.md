<div align="center">

# 🤖 HelpDeskAi
### AI-Powered Customer Support Ticket Orchestrator & Auto-Responder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-F9D371.svg)](https://huggingface.co/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF4b4b?style=flat&logo=chroma&logoColor=white)](https://www.trychroma.com/)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)](https://www.langchain.com/)

</div>

---

## 📖 Project Abstract

**HelpDeskAi** is an intelligent, edge-capable customer support orchestration pipeline built for the Synthetix 4.0 Hackathon. 

The system relies on a hybrid AI architecture that combines the rapid inference of specialized Deep Learning models with the nuanced reasoning of Large Language Models. At its core sits a custom **Multi-Task `DistilBERT` classification model** trained to simultaneously predict a support ticket's category and its relative urgency. When a ticket arrives, the framework routes it through the neural network while simultaneously vectorizing its contents to perform a semantic search against a local **ChromaDB Knowledge Base**. In the final step, the system uses **Retrieval-Augmented Generation (RAG)** via LangChain and the Groq API (Llama3-8b) to synthesize a highly accurate, grounded draft response for human agents.

## ✨ Key Features (100% Core Requirements Complete)

*   🧠 **Dual-Headed DistilBERT Architecture:** Utilizes a PyTorch Multi-Task learning model that outputs both `Category` and `Urgency` in a single forward pass, doubling inference speed.
*   📐 **Confidence Math (Softmax):** Converts raw Neural Network logits into clean, human-readable confidence percentages.
*   📚 **Omni-Format Knowledge Base (RAG):** Automatically parses, chunks, and locally embeds company policies from `.txt`, `.md`, and **`.pdf`** formats.
*   📝 **Anti-Hallucination Guardrails:** Employs a zero-tolerance Llama3 prompt that forces the LLM to output *"I need more clarification to resolve this."* if the Vector DB cannot provide relevant KB citations.
*   ⚙️ **Fully Automated FastAPI Backend:** Exposes a robust JSON endpoint that accepts ticket metadata (`subject`, `description`, `timestamp`, `channel`) and returns classifications, a 1-sentence AI summary, the drafted reply, and citation sources.

---

## 🏗️ Architecture Workflow

Here is exactly how a ticket flows through the HelpDeskAi ecosystem:

```mermaid
graph TD
    classDef step fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:white,padding:10px;
    classDef db fill:#020617,stroke:#10B981,stroke-width:2px,color:white;
    classDef output fill:#312E81,stroke:#8B5CF6,stroke-width:2px,color:white;

    Ticket["📥 1. Inbound Ticket<br>(Subject, Description, Metadata)"]:::step --> API["⚙️ FastAPI Orchestrator"]:::step
    
    API --> Tokenizer["🔤 2. Padding/Truncation<br>(Max 128 Tokens)"]:::step
    Tokenizer --> DL["🧠 3. Multi-Task DistilBERT<br>(Local Deep Learning)"]:::step
    
    DL --> Cat["Category Projection"]:::output
    DL --> Urg["Urgency Projection"]:::output
    DL --> Math["Softmax Confidence %"]:::output
    
    API --> Embed["🔢 4. SentenceTransformer<br>(all-MiniLM-L6-v2)"]:::step
    Embed --> ChromaDB[("📚 5. ChromaDB Vector Store<br>Top-K Semantic Search")]:::db
    
    ChromaDB --> Sources["Retrieved Context & Metadata (Citations)"]:::output
    Sources --> Groq["💬 6. Groq LLM Generation<br>(Llama3-8b via LangChain)"]:::step
    
    Groq --> Summary["AI Summary"]:::output
    Groq --> Reply["Grounded Draft Reply"]:::output
```

---

## 📊 Data Engineering & Datasets

To ensure the classifier models were robust and battle-tested, we abandoned synthetic mock data and heavily engineered real-world customer support structures.

**The Multilingual Dataset Migration**
We integrated the comprehensive `Tobi-Bueck/customer-support-tickets` dataset dynamically hosted on HuggingFace. 
1.  **Extraction:** Created `preprocess_hf.py` to securely download data partitions via the HuggingFace `datasets` library.
2.  **Transformation:** Consolidated subjects and descriptive bodies into unified contexts, mapped exact taxonomic categories (`Incident`, `Problem`, `Change`, `Request`), and procedurally synthesized Urgency ratings based on internal ITIL matrices.
3.  **Loading:** The resultant 5,000+ ticket subset was then fed into our PyTorch `train_classifier.py` pipeline for multi-epoch, localized CPU fine-tuning.

---
<div align="center">
<i>Built for the Synthetix 4.0 Hackathon</i>
</div>
