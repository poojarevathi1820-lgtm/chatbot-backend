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
from langchain_core.runnables import RunnableBranch, RunnableLambda
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

# --- 2. API DATA SCHEMAS ---

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

# --- 3. HELPER FUNCTIONS ---

def format_search_results(results_json: str) -> str:
    """
    Takes the JSON from Supabase and formats it into clean, professional HTML.
    This function has the final, correct logic.
    """
    try:
        results = json.loads(results_json)
        if "error" in results or not results:
            return "<p>I'm sorry, I couldn't find any cases matching your search. Please try another model!</p>"

        # REQUIREMENT 3: Limit the number of results to 3
        results_to_display = results[:3]
        
        html_responses = ["<p>Here are the top results I found:</p>"]

        for case in results_to_display:
            case_id = case.get('id', '')
            name = case.get('name', 'N/A')
            cost = case.get('cost', 'N/A')
            image_url = case.get('image_url', '')

            # This is the perfect, clean HTML that will be sent to the user.
            response_html = (
                f"<div style='margin-top: 15px;'>"
                f"<p style='margin:0; font-weight: bold;'>{name}</p>"
                f"<p style='margin:0;'><b>Price:</b> ${cost}</p>"
                f'<a href="product.html?id={case_id}" target="_blank" title="View Details for {name}">'
                f'<img src="{image_url}" alt="{name}" style="max-width: 100%; height: auto; border-radius: 10px; margin-top: 5px; cursor: pointer;" />'
                f'</a>'
                f"</div>"
            )
            html_responses.append(response_html)
        
        # REQUIREMENT 4: If there were more results than we displayed, add a "See More" link.
        if len(results) > 3:
            # You can change 'cases.html' to whatever your main product page is.
            see_more_link = "<p style='text-align:center; margin-top: 20px;'><a href='cases.html' target='_blank'>See More Cases</a></p>"
            html_responses.append(see_more_link)

        # Join all parts with a simple newline, as they are self-contained HTML blocks.
        return "".join(html_responses)

    except json.JSONDecodeError:
        return "<p>Error: I had trouble processing the search results.</p>"

def search_cases_in_supabase(search_term: str) -> str:
    if not search_term.strip():
        return json.dumps([])
    print(f"--- DB Query: Searching for cleaned term '{search_term}' ---")
    try:
        brand_filter = f"brand.ilike.%{search_term}%"
        model_filter = f"model.ilike.%{search_term}%"
        name_filter = f"name.ilike.%{search_term}%"
        combined_filters = f"{brand_filter},{model_filter},{name_filter}"

        # Fetch up to 10 results so we know if we need to show the "See More" link.
        query = supabase.table("cases").select(
            "id, name, model, brand, color, material, image_url, cost"
        ).or_(combined_filters).limit(10)
        
        response = query.execute()
        return json.dumps(response.data) if response.data else json.dumps([])
    except Exception as e:
        print(f"!!! DATABASE ERROR: {e} !!!")
        return json.dumps({"error": "Failed to query the database."})

# --- 4. LANGCHAIN PIPELINE ---

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)

# THIS PROMPT IS NOW COMPLETELY IRRELEVANT, AS WE BYPASS THE AI FOR FORMATTING.
# The chain's final step is our Python function, not the AI.

keyword_extraction_chain = (
    ChatPromptTemplate.from_template(
        "Extract only the product keywords from this message: {message}. Keywords:"
    )
    | llm
    | StrOutputParser()
)

# --- THIS IS THE FINAL, CORRECT CHAIN ---
# IT DOES NOT USE THE AI TO FORMAT THE FINAL RESPONSE.
rag_chain = (
    # Step 1: Get the user's message
    {"message": lambda x: x["message"]}
    # Step 2: Extract the keywords
    | {"cleaned_term": keyword_extraction_chain}
    # Step 3: Search the database with the keywords
    | RunnableLambda(lambda x: search_cases_in_supabase(search_term=x["cleaned_term"]))
    # Step 4: Format the results into perfect HTML using our Python function
    | RunnableLambda(format_search_results)
)

general_chat_chain = (
    ChatPromptTemplate.from_template("Respond warmly: {message}") | llm | StrOutputParser()
)

main_pipeline = (
    {"message": lambda x: x["message"]}
    | {
        "intent": ChatPromptTemplate.from_template(
            "Classify intent as 'product_search' or 'general_chat'. Message: {message}"
        ) | llm | StrOutputParser(),
        "message": lambda x: x["message"],
    }
    | RunnableBranch(
        (lambda x: "product_search" in x["intent"], rag_chain),
        general_chat_chain,
    )
)

# --- 5. FASTAPI WEB SERVER (No changes needed here) ---
origins = ["*"]
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
    uvicorn.run("bot:app", host="0.0.0.0", port=8000, reload=True)