# main.py

import os
import json
from typing import List, Optional, Dict
from typing_extensions import TypedDict

# --- Library Imports ---
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase.client import Client, create_client

# --- LangChain & LangGraph Imports ---
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.managed import ThreadManaged

# --- 1. INITIALIZATION AND CONFIGURATION ---

load_dotenv()

# Initialize Google AI Model
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY must be set in the .env file.")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)

# Initialize Supabase Client for live queries
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not all([supabase_url, supabase_key]):
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the .env file.")
try:
    supabase: Client = create_client(supabase_url, supabase_key)
    print("--- Successfully connected to Supabase for live queries. ---")
except Exception as e:
    print(f"--- Error connecting to Supabase: {e} ---")
    exit()


# --- 2. DESIGN PROMPTS ---

# Prompt for the Router: This decides if a question is about products or general chat.
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at routing a user question.
Based on the user's question, classify it as either 'product_related' or 'general_chat'.
For example:
- "show me vivo cases" -> product_related
- "hi" -> general_chat
- "do you have anything for an iPhone 15?" -> product_related
- "thank you" -> general_chat

Do not respond with more than one word."""),
    ("human", "{question}")
])

# Prompt for the Product Formatter: This takes the raw JSON from Supabase and makes it beautiful for the user.
product_formatter_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful e-commerce assistant. Your goal is to format product data from a database into a clear, user-friendly response.

**CRITICAL INSTRUCTIONS:**
1.  Take the JSON data provided and format each product found.
2.  For each product, you MUST include:
    - The product `name` in **bold**.
    - The `brand` and `model` it is for.
    - The `cost` on a new line, formatted as **Price:** ₹[cost].
    - The `material` on a new line, formatted as **Material:** [material].
    - **Most Importantly:** The image, formatted as a Markdown image link: `![Product Name](image_url)`.
3.  If the JSON data is an empty list `[]`, you MUST respond with: "I'm sorry, I couldn't find any cases matching your search."
4.  Separate multiple products with a horizontal line (`---`).

Here is the JSON data from the database:"""),
    ("human", "{product_json}")
])

# Prompt for General Chat: This handles conversational questions with memory.
general_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and helpful e-commerce assistant. Respond to the user in a conversational manner, remembering the previous parts of the conversation."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# --- 3. DEFINE THE LANGGRAPH STATE & NODES ---

class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    chat_history: list[BaseMessage]
    products: Optional[List[Dict]] # To store live product data from Supabase

# --- Define the Nodes (the steps in our graph) ---

def search_supabase_node(state: GraphState):
    """Node that performs a LIVE query to Supabase to find products."""
    question = state["question"]
    print(f"--- [NODE] Searching Supabase live for: '{question}' ---")
    
    try:
        # This flexible filter searches the user's question across three different columns.
        filter_string = f"model.ilike.%{question}%,brand.ilike.%{question}%,name.ilike.%{question}%"
        response = supabase.table("cases").select(
            "name, model, brand, color, material, image_url, cost"
        ).or_(filter_string).limit(5).execute()
        
        products = response.data or []
        print(f"--- [NODE] Supabase query complete. Found {len(products)} products. ---")
        return {"products": products}
    except Exception as e:
        print(f"--- [NODE] Error during Supabase query: {e} ---")
        return {"products": []} # Return empty list on error to handle gracefully

def format_product_response_node(state: GraphState):
    """Node that uses an LLM to format the Supabase JSON into a user-friendly response."""
    print("--- [NODE] Formatting product JSON into a final response. ---")
    products = state["products"]
    
    formatter_chain = product_formatter_prompt | llm
    
    product_json_string = json.dumps(products, indent=2)
    
    response = formatter_chain.invoke({"product_json": product_json_string})
    print("--- [NODE] Formatting complete. ---")
    return {"question": response.content} # The final formatted answer

def general_chat_node(state: GraphState):
    """Node that handles general conversation."""
    print("--- [NODE] Generating general chat answer. ---")
    question = state["question"]
    chat_history = state["chat_history"]
    
    chat_chain = general_chat_prompt | llm
    response = chat_chain.invoke({"question": question, "chat_history": chat_history})
    return {"question": response.content}

def route_question(state: GraphState):
    """Node that routes the question to the appropriate path."""
    print("--- [ROUTER] Deciding path... ---")
    question = state["question"]
    router_chain = router_prompt | llm
    decision = router_chain.invoke({"question": question})
    
    if "product" in decision.content.lower():
        print("--- [ROUTER] Decision: Product related. Routing to live Supabase search. ---")
        return "search_supabase"
    else:
        print("--- [ROUTER] Decision: General chat. Routing to conversational chain. ---")
        return "general_chat"

# --- 4. BUILD THE LANGGRAPH GRAPH ---

workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("search_supabase", search_supabase_node)
workflow.add_node("format_product_response", format_product_response_node)
workflow.add_node("general_chat", general_chat_node)

# Define the graph's entry point and conditional routing
workflow.add_conditional_edges(
    "__start__",
    route_question,
    {
        "search_supabase": "search_supabase",
        "general_chat": "general_chat",
    },
)

# Define the connections between nodes
workflow.add_edge("search_supabase", "format_product_response")
workflow.add_edge("format_product_response", END)
workflow.add_edge("general_chat", END)

# Compile the graph and add memory management
memory = ThreadManaged(graph=workflow.compile(), thread_key="thread_id")

# --- 5. FASTAPI WEB SERVER ---

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

app = FastAPI(
    title="LangGraph Live E-commerce Chatbot API",
    description="A chatbot that queries Supabase in real-time to answer product questions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handles chat messages, maintains conversation history, and routes to the correct skill."""
    print(f"\n--- [REQUEST] Received message: '{request.message}' for thread: {request.thread_id} ---")
    try:
        config = {"thread_id": request.thread_id}
        input_message = HumanMessage(content=request.message)
        
        final_answer = ""
        # Stream the graph's execution to get the final result
        for chunk in memory.stream([input_message], config=config):
            # The final answer is in the 'question' key of the last state update
            final_answer = chunk.get("question", "")

        print(f"--- [RESPONSE] Sending final answer to frontend. ---")
        return ChatResponse(reply=final_answer)

    except Exception as e:
        print(f"--- [ERROR] An error occurred during graph invocation: {e} ---")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)