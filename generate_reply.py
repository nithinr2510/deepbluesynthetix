import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

def draft_support_reply(ticket_text: str, retrieved_context: list[str]) -> str:
    """
    Drafts a support reply based on the user's ticket and the retrieved context.
    Uses ChatGroq (llama3-8b-8192) and explicitly avoids hallucination.
    """
    
    # Ensure GROQ_API_KEY is available (if not, it will fail when calling the API)
    if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_api_key_here":
        return "Error: GROQ_API_KEY is not configured properly in the environment."

    # Initialize the Groq model
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0
    )
    
    # Create the prompt template with strict instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a support agent. Using ONLY the provided retrieved context, draft a reply. If the context does not contain the answer to the user's issue, output EXACTLY: \"I need more clarification to resolve this.\" Do not make up information.\n\nContext:\n{context}"),
        ("human", "User Issue: {issue}")
    ])
    
    # Format the context list into a single string
    context_str = "\n\n---\n\n".join(retrieved_context) if retrieved_context else "No context found."
    
    # Build the LCEL pipeline
    chain = prompt | llm | StrOutputParser()
    
    # Invoke the chain
    response = chain.invoke({
        "context": context_str,
        "issue": ticket_text
    })
    
    return response

if __name__ == '__main__':
    # Add a dummy API key for testing if it's missing just so LangChain initialization doesn't immediately crash
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = "mock_key_for_testing"

    print("=== Test 1: Relevant Context ===")
    mock_ticket = "How do I reset my password?"
    mock_context = [
        "How to Reset Your Password\n\n"
        "If you forgot your password, go to the login page and click 'Forgot Password'. "
        "Enter your registered email address, and we will send you a reset link. "
        "Click the link and follow the instructions to create a new password."
    ]
    
    # To run this successfully, you need a valid GROQ_API_KEY.
    # We use a try-except here to gracefully handle the fact that we might be using the dummy key.
    try:
        reply_1 = draft_support_reply(mock_ticket, mock_context)
        print(f"Ticket: {mock_ticket}")
        print(f"Reply:\n{reply_1}\n")
    except Exception as e:
        print(f"Test 1 failed (likely due to invalid API key): {e}\n")

    print("=== Test 2: Unrelated Ticket with Empty Context ===")
    mock_ticket_unrelated = "How do I bake a cake?"
    mock_context_empty = []
    
    try:
        reply_2 = draft_support_reply(mock_ticket_unrelated, mock_context_empty)
        print(f"Ticket: {mock_ticket_unrelated}")
        print(f"Reply:\n{reply_2}\n")
    except Exception as e:
        print(f"Test 2 failed: {e}\n")
