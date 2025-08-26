import os
import json
from typing import Optional

# --- Library Imports ---
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase.client import Client, create_client

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. INITIALIZATION AND CONFIGURATION ---

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not all([supabase_url, supabase_key, google_api_key]):
    raise ValueError("Supabase URL/Key and Google API Key must be set in the .env file.")

try:
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Successfully connected to Supabase.")
except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    exit()

# --- 2. API DATA SCHEMAS (PYDANTIC MODELS) ---

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

# --- 3. DATABASE LOGIC ---

def search_cases_in_supabase(search_term: str) -> str:
    """
    Constructs and executes a broad 'OR' query using a CLEANED search term.
    """
    # If the cleaned search term is empty, don't bother searching.
    if not search_term.strip():
        return json.dumps([])
        
    print(f"--- DB Query: Searching for cleaned term '{search_term}' in brand, model, or name columns ---")
    try:
        brand_filter = f"brand.ilike.%{search_term}%"
        model_filter = f"model.ilike.%{search_term}%"
        name_filter = f"name.ilike.%{search_term}%"
        combined_filters = f"{brand_filter},{model_filter},{name_filter}"

        query = supabase.table("cases").select(
            "name, model, brand, color, material, image_url, cost"
        ).or_(combined_filters)
        
        response = query.limit(5).execute()
        
        if not response.data:
            print("--- DB Query: Found 0 results. ---")
            return json.dumps([])
        
        print(f"--- DB Query: Found {len(response.data)} result(s). ---")
        return json.dumps(response.data)

    except Exception as e:
        print(f"!!!!!!!!!!!!!!!! AN ERROR OCCURRED: {e} !!!!!!!!!!!!!!!!")
        return json.dumps({"error": "Failed to query the database."})

# --- 4. LANGCHAIN RAG PIPELINE SETUP ---

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

# --- NLP STEP 1: Intent Classification ---
intent_classifier_prompt = ChatPromptTemplate.from_template(
    """Given the user's message, classify the intent as 'product_search' or 'general_chat'.
    User Message: "{message}"
    Intent:"""
)
intent_classifier_chain = intent_classifier_prompt | llm | StrOutputParser()

# --- NLP STEP 2: KEYWORD EXTRACTION (THE NEW, CRITICAL STEP) ---
keyword_extraction_prompt = ChatPromptTemplate.from_template(
    """From the following user message, extract ONLY the essential product keywords like the phone brand and model.
    Remove all conversational filler words like "show me", "cases for", "a", "an", "I want".
    For example:
    - "show me vivo v40 cases" becomes "vivo v40"
    - "do you have anything for an oppo 16" becomes "oppo 16"
    - "iphone 17 pro" becomes "iphone 17 pro"

    User Message: "{message}"
    Keywords:"""
)
keyword_extraction_chain = keyword_extraction_prompt | llm | StrOutputParser()


# --- RAG CHAIN: For handling product searches ---
rag_prompt_template = """You are an expert e-commerce assistant for TravaCaseAI.
Answer the user's question based ONLY on the following search results.
If the search results are empty, politely inform the user that you couldn't find any matching cases.
DO NOT make up any products.
SEARCH RESULTS: {context}
USER'S QUESTION: {question}
YOUR FRIENDLY RESPONSE (use Markdown for formatting, including images):
"""
rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

rag_chain = (
    {
        # We pass the original message through to use as the "question" at the end.
        "question": RunnablePassthrough(),
        # We use the new keyword chain to create a clean search term.
        "cleaned_term": keyword_extraction_chain,
    }
    | RunnableLambda(lambda x: {
        # We search the database using ONLY the cleaned term.
        "context": search_cases_in_supabase(search_term=x["cleaned_term"]),
        "question": x["question"]["message"] # We need the original message for the final prompt
    })
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- GENERAL CHAT CHAIN ---
general_chat_prompt = ChatPromptTemplate.from_template(
    "You are a friendly assistant. Respond warmly to the user's message.\nUser Message: {message}\nYour Response:"
)
general_chat_chain = general_chat_prompt | llm | StrOutputParser()

# --- Full Pipeline with Router ---
main_pipeline = {
    # The input to the pipeline is a dictionary: {"message": "user's text"}
    "intent": intent_classifier_chain,
    "message": lambda x: x["message"]
} | RunnableBranch(
    (lambda x: "product_search" in x["intent"], rag_chain),
    general_chat_chain
)

# --- 5. FASTAPI WEB SERVER ---

app = FastAPI(title="Supabase E-commerce Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="API Status Check")
def read_root(): return {"status": "API is running"}

@app.post("/chat", response_model=ChatResponse, summary="Handle Chat Messages")
async def chat(request: ChatRequest):
    print(f"Received message: '{request.message}'")
    try:
        # The pipeline expects a dictionary as input
        response = main_pipeline.invoke({"message": request.message})
        return ChatResponse(reply=response)
    except Exception as e:
        print(f"Error during pipeline invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("bot:app", host="0.0.0.0", port=8000, reload=True)