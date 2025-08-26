import os
import json
from typing import Optional
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase.client import Client, create_client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. INITIALIZATION AND CONFIGURATION (No changes needed) ---
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

# --- 2. API DATA SCHEMAS (No changes needed) ---
class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

# --- 3. HELPER FUNCTIONS ---

def format_search_results(results_json: str) -> str:
    # No changes needed in this function
    try:
        results = json.loads(results_json)
        if "error" in results or not results:
            return "I'm sorry, I couldn't find any cases matching your search. Please try another model!"
        results_to_display = results[:3]
        html_responses = ["<p>Here are the top results I found:</p>"]
        for case in results_to_display:
            case_id, name, cost, image_url = case.get('id', ''), case.get('name', 'N/A'), case.get('cost', 'N/A'), case.get('image_url', '')
            response_html = (
                f"<div style='margin-top: 15px;'><p style='margin:0; font-weight: bold;'>{name}</p>"
                f"<p style='margin:0;'><b>Price:</b> ${cost}</p><a href='product.html?id={case_id}' target='_blank' title='View Details for {name}'>"
                f"<img src='{image_url}' alt='{name}' style='max-width: 100%; height: auto; border-radius: 10px; margin-top: 5px; cursor: pointer;' /></a></div>"
            )
            html_responses.append(response_html)
        if len(results) > 3:
            see_more_link = "<p style='text-align:center; margin-top: 20px;'><a href='cases.html' target='_blank'>See More Cases</a></p>"
            html_responses.append(see_more_link)
        return "".join(html_responses)
    except json.JSONDecodeError:
        return "<p>Error: I had trouble processing the search results.</p>"

# THIS IS THE CORRECTED AND IMPROVED FUNCTION
def search_cases_in_supabase(search_term: str) -> str:
    """
    Splits the search term into keywords and searches for rows that match ALL keywords.
    """
    if not search_term.strip():
        return json.dumps([])
    
    # 1. Split the search term into individual words
    keywords = search_term.split()
    print(f"--- DB Query: Searching for keywords {keywords} ---")

    try:
        query = supabase.table("cases").select(
            "id, name, model, brand, color, material, image_url, cost"
        )

        # 2. Chain `ilike` filters for each keyword to find matches
        # This will find products where the name contains "vivo" AND "v40"
        for keyword in keywords:
            query = query.ilike('name', f'%{keyword}%')

        response = query.limit(10).execute()
        return json.dumps(response.data) if response.data else json.dumps([])
        
    except Exception as e:
        print(f"!!! DATABASE ERROR: {e} !!!")
        return json.dumps({"error": "Failed to query the database."})

# --- 4. LANGCHAIN PIPELINE (No changes needed) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)

keyword_extraction_chain = (
    ChatPromptTemplate.from_template("Extract only the product keywords from this message: {message}. Keywords:")
    | llm
    | StrOutputParser()
)
rag_chain = (
    {"message": lambda x: x["message"]}
    | {"cleaned_term": keyword_extraction_chain}
    | RunnableLambda(lambda x: search_cases_in_supabase(search_term=x["cleaned_term"]))
    | RunnableLambda(format_search_results)
)
general_chat_chain = (
    ChatPromptTemplate.from_template("Respond warmly: {message}") | llm | StrOutputParser()
)
main_pipeline = (
    {"message": lambda x: x["message"]}
    | {"intent": ChatPromptTemplate.from_template("Classify intent as 'product_search' or 'general_chat'. Message: {message}") | llm | StrOutputParser(), "message": lambda x: x["message"]}
    | RunnableBranch((lambda x: "product_search" in x["intent"], rag_chain), general_chat_chain)
)

# --- 5. FASTAPI WEB SERVER (No changes needed) ---
app = FastAPI(title="Supabase E-commerce Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", summary="API Status Check")
def read_root(): return {"status": "API is running"}

@app.get("/product/{product_id}", summary="Get Single Product Details")
async def get_product(product_id: str):
    try:
        response = supabase.table("cases").select("*").eq("id", product_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Product not found")
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse, summary="Handle Chat Messages")
async def chat(request: ChatRequest):
    print(f"Received message: '{request.message}'")
    try:
        response = main_pipeline.invoke({"message": request.message})
        return ChatResponse(reply=response)
    except Exception as e:
        print(f"Error during pipeline invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("bot1:app", host="0.0.0.0", port=8000, reload=True)