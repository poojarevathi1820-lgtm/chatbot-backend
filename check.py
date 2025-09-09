import os
import json
import re
from typing import Optional, List, Dict, Any

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

# ... (your existing FAQS, FAQ_KEYWORDS, FAQ_PHRASES, detect_static_intent remain the same) ...

FAQS = {
    "best material": "The best material depends on your needs. TPU and silicone offer flexibility and grip, polycarbonate is very durable, and leather is stylish.",
    "protect from drops": "Yes, our cases are designed to protect your phone from everyday bumps and drops. Rugged and shockproof models offer the highest protection.",
    "wireless charging": "Most of our slim and regular-fit cases are compatible with wireless charging. Very thick or metal cases may interfere.",
    "clean case": "Gently scrub with warm water and dish soap. For tough stains, use a baking soda paste. Remove your phone before cleaning.",
    "yellow case": "Yellowing is due to UV and skin oils. Cleaning with baking soda may help, but regular cleaning is best to prevent it.",
    "warranty": "We offer a one-year warranty covering manufacturing defects. Contact support with proof of purchase for claims.",
    "how to order": "Ordering from Travacase is easy! Just browse our collection, select your desired case, add it to your cart, and proceed to checkout. Follow the prompts to enter your shipping and payment details, and confirm your order. You'll receive an email confirmation shortly after!",
    "my orders": "Here are your order details:"
}

FAQ_KEYWORDS = {
    "material": "best material",
    "drop": "protect from drops",
    "drops": "protect from drops",
    "wireless": "wireless charging",
    "charging": "wireless charging",
    "clean": "clean case",
    "yellow": "yellow case",
    "warranty": "warranty",
    "order": "how to order",
    "ordering": "how to order",
    "buy": "how to order",
    "purchase": "how to order",
    "my orders": "my orders",
    "my order": "my orders",
    "previous orders": "my orders",
    "past orders": "my orders"
}

FAQ_PHRASES = [
    ("best material", ["best material", "what material", "which material"]),
    ("protect from drops", ["protect from drops", "drop protection", "protect my phone", "shockproof"]),
    ("wireless charging", ["wireless charging", "charge wirelessly", "wireless charger", "support wireless"]),
    ("clean case", ["clean case", "how to clean", "clean my case", "cleaning case"]),
    ("yellow case", ["yellow case", "case turned yellow", "clear case yellow", "yellowing"]),
    ("warranty", ["warranty", "guarantee", "return policy", "replace my case"]),
    ("how to order", ["how to order", "how can i order", "place an order", "buy a case", "purchase a case"]),
    ("my orders", ["my orders", "my order details", "show my orders", "what are my past orders", "previous orders"])
]

def detect_static_intent(message: str) -> Optional[str]:
    msg = message.lower().strip()
    tokens = set(re.findall(r'\w+', msg))

    new_arrivals_patterns = [
        r"\bnew arrivals\b", r"\blatest cases\b", r"\bwhat'?s new\b",
        r"\brecent cases\b", r"\bjust arrived\b", r"\bnewest cases\b",
        r"\bshow me new arrivals\b", r"\bsee new arrivals\b", r"\bany new arrivals\b",
        r"\bcan i see new arrivals\b", r"\bdo you have new arrivals\b", r"\bwhat are your new arrivals\b"
    ]
    for pattern in new_arrivals_patterns:
        if re.search(pattern, msg):
            return "new_arrivals"

    top_selling_patterns = [
        r"\btop selling\b", r"\bbest sellers\b", r"\bmost popular\b",
        r"\bpopular cases\b", r"\bhot cases\b", r"\bshow me top selling\b",
        r"\bsee top selling\b", r"\bany top selling\b"
    ]
    for pattern in top_selling_patterns:
        if re.search(pattern, msg):
            return "top_selling" # This will now correctly trigger the top_selling path

    my_orders_patterns = [
        r"\bmy orders\b", r"\bmy past orders\b", r"\bmy previous orders\b",
        r"\bwhere is my order\b", r"\bcheck my order\b", r"\bwhat did i order\b"
    ]
    for pattern in my_orders_patterns:
        if re.search(pattern, msg):
            return "my_orders_request"

    for faq_key, phrases in FAQ_PHRASES:
        for phrase in phrases:
            if phrase in msg:
                if faq_key == "how to order":
                    return "how_to_order_faq"
                return "faq"
    for keyword in FAQ_KEYWORDS:
        if keyword in tokens or keyword + "s" in tokens:
            if FAQ_KEYWORDS[keyword] == "how to order":
                return "how_to_order_faq"
            return "faq"

    return None

def format_new_arrivals(items: List[Dict[str, Any]]):
    html = "<h3>Here are our latest arrivals! 🎉</h3><div style='display:flex;flex-wrap:wrap;gap:16px;'>"
    if not items:
        return "<p>Sorry, no new arrivals found at the moment.</p>"
    for item in items:
        html += (
            f"<div style='width:150px;text-align:center;'>"
            f"<img src='{item.get('image_url', '')}' alt='{item.get('name', '')}' style='width:100%;border-radius:8px;'/>"
            f"<div style='margin-top:8px;font-weight:bold;'>{item.get('name', 'N/A')}</div>"
            f"</div>"
        )
    html += "</div><p>Let me know if you'd like more details or want to order any of these!</p>"
    return html

def format_top_selling(items: List[Dict[str, Any]]):
    html = "<h3>Check out our top-selling cases! 🔥</h3><div style='display:flex;flex-wrap:wrap;gap:16px;'>"
    if not items:
        return "<p>Sorry, no top-selling cases found at the moment.</p>"
    for item in items:
        # Assuming your top_selling table has 'name', 'model', 'cost', 'image_url'
        # Adjust 'model' to 'name' if you only have a single 'name' field that includes the model info
        # The 'id' for the link will not be present in the top_selling table, so we'll remove it from the link for now.
        # If you need a link to a product page, the 'top_selling' table would need to store the 'id' from the main 'cases' table.
        name = item.get('name', 'N/A')
        cost = item.get('cost', 'N/A')
        image_url = item.get('image_url', '')

        html += (
            f"<div style='width:150px;text-align:center;'>"
            f"<img src='{image_url}' alt='{name}' style='width:100%;border-radius:8px;'/>"
            f"<div style='margin-top:8px;font-weight:bold;'>{name}</div>"
            f"<div style='font-size:0.9em;'>₹{cost}</div>"
            f"</div>"
        )
    html += "</div><p>Grab them before they're gone!</p>"
    return html


def match_faq(user_message: str) -> Optional[str]:
    msg = user_message.lower()
    tokens = set(re.findall(r'\w+', msg))
    for faq_key, phrases in FAQ_PHRASES:
        for phrase in phrases:
            if phrase in msg:
                return f"<b>FAQ:</b> {FAQS[faq_key]}"
    for keyword, faq_key in FAQ_KEYWORDS.items():
        if keyword in tokens or keyword + "s" in tokens:
            return f"<b>FAQ:</b> {FAQS[faq_key]}"
    return None

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

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

def format_search_results(results_json: str) -> str:
    try:
        results = json.loads(results_json)
        if "error" in results or not results:
            return "<p>Sorry, I couldn't find any cases matching your search. Please try another model!</p>"
        results_to_display = results[:3]
        html_responses = ["<p>Here are the top results I found:</p>"]
        for case in results_to_display:
            case_id = case.get('id', '')
            name = case.get('name', 'N/A')
            cost = case.get('cost', 'N/A')
            image_url = case.get('image_url', '')
            response_html = (
                f"<div style='margin-top: 15px;'>"
                f"<p style='margin:0; font-weight: bold;'>{name}</p>"
                f"<p style='margin:0;'><b>Price:</b> ₹ {cost}</p>"
                f'<a href="product.html?id={case_id}" target="_blank" title="View Details for {name}">'
                f'<img src="{image_url}" alt="{name}" style="max-width: 100%; height: auto; border-radius: 10px; margin-top: 5px; cursor: pointer;" />'
                f'</a>'
                f"</div>"
            )
            html_responses.append(response_html)
        if len(results) > 3:
            see_more_link = "<p style='text-align:center; margin-top: 20px;'><a href='cases.html' target='_blank'>See More Cases</a></p>"
            html_responses.append(see_more_link)
        return "".join(html_responses)
    except json.JSONDecodeError:
        return "<p>Error: I had trouble processing the search results.</p>"

def parse_brand_model(extracted: str) -> dict:
    brand = ''
    model = ''
    for part in extracted.split(','):
        part = part.strip()
        if part.lower().startswith('brand:'):
            brand = part.split(':', 1)[1].strip()
        elif part.lower().startswith('model:'):
            model = part.split(':', 1)[1].strip()
    return {'brand': brand, 'model': model}

def search_cases_in_supabase(brand: str, model: str) -> str:
    try:
        query = supabase.table("cases").select(
            "id, name, model, brand, color, material, image_url, cost"
        )
        if brand and model:
            query = query.ilike("brand", f"%{brand}%").ilike("model", f"%{model}%")
        elif brand:
            query = query.ilike("brand", f"%{brand}%")
        elif model:
            query = query.ilike("model", f"%{model}%")
        else:
            return json.dumps([])
        query = query.limit(10)
        response = query.execute()
        return json.dumps(response.data) if response.data else json.dumps([])
    except Exception as e:
        print(f"!!! DATABASE ERROR: {e} !!!")
        return json.dumps({"error": "Failed to query the database."})

async def get_top_selling_cases() -> List[Dict[str, Any]]:
    """
    Fetches top selling cases from the dedicated 'top_selling' Supabase table.
    Assumes the 'top_selling' table has 'name', 'model', 'cost', 'image_url' columns.
    """
    try:
        # Query the 'top_selling' table directly
        response = supabase.table("top_selling").select("name, model, cost, image_url").limit(4).execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching top selling cases from 'top_selling' table: {e}")
        return []

async def get_new_arrivals_cases() -> List[Dict[str, Any]]:
    """
    Fetches new arrivals from the 'cases' table.
    """
    try:
        response = supabase.table("cases").select("name, image_url").order("created_at", desc=True).limit(4).execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching new arrivals cases: {e}")
        return []

async def get_user_orders(user_id: str) -> str:
    try:
        response = supabase.table("orders").select(
            "*, profiles!orders_user_id_fkey(name)"
        ).eq("user_id", user_id).execute()

        if not response.data:
            return "<p>I couldn't find any orders for your user ID. Please ensure you are logged in or contact support if you believe this is an error.</p>"

        user_name = "Customer"
        if response.data[0] and 'profiles' in response.data[0] and response.data[0]['profiles']:
            user_name = response.data[0]['profiles']['name']

        html_responses = [f"<h3>Here are the order details for {user_name}:</h3>"]
        for order in response.data:
            order_id = order.get('id', 'N/A')
            order_date = order.get('created_at', 'N/A').split('T')[0]
            items = order.get('items', [])
            total_cost = order.get('total_cost', 'N/A')

            item_details = "<ul>"
            for item in items:
                item_name = item.get('name', 'N/A')
                item_qty = item.get('quantity', 1)
                item_price = item.get('price', 'N/A')
                item_details += f"<li>{item_name} (x{item_qty}) - &#8377;{item_price} each</li>"
            item_details += "</ul>"

            response_html = (
                f"<div style='border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px;'>"
                f"<p style='margin:0; font-weight: bold;'>Order ID: {order_id}</p>"
                f"<p style='margin:0;'>Order Date: {order_date}</p>"
                f"<p style='margin:0;'>Items:</p>{item_details}"
                f"<p style='margin:0;'><b>Total Cost:</b> &#8377;{total_cost}</p>"
                f"</div>"
            )
            html_responses.append(response_html)

        return "".join(html_responses)

    except Exception as e:
        print(f"Error fetching user orders: {e}")
        return "<p>Sorry, I encountered an error while trying to fetch your orders. Please try again later.</p>"


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)

keyword_extraction_chain = (
    ChatPromptTemplate.from_template(
        "Extract the product brand and model from this message: {message}. "
        "Return as: brand: <brand>, model: <model>. If either is missing, leave blank."
    )
    | llm
    | StrOutputParser()
)

rag_chain = (
    {"message": lambda x: x["message"]}
    | {"extracted": keyword_extraction_chain}
    | RunnableLambda(lambda x: parse_brand_model(x["extracted"]))
    | RunnableLambda(lambda x: search_cases_in_supabase(brand=x["brand"], model=x["model"]))
    | RunnableLambda(format_search_results)
)

general_chat_prompt = ChatPromptTemplate.from_template("You are a helpful and friendly chatbot for Travacase. Respond warmly and assist the user. Message: {message}")

general_chat_chain = (
    general_chat_prompt | llm | StrOutputParser()
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

app = FastAPI(title="Supabase E-commerce Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be cautious with "*" in production; specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="API Status Check")
def read_root():
    return {"status": "API is running"}

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
    print(f"Received message: '{request.message}' for session '{request.session_id}' with user_id: '{request.user_id}'")
    try:
        static_intent = detect_static_intent(request.message)

        if static_intent == "new_arrivals":
            new_arrivals_cases = await get_new_arrivals_cases()
            return ChatResponse(reply=format_new_arrivals(new_arrivals_cases))
        elif static_intent == "top_selling":
            # This is the correct path for top_selling
            top_selling_cases = await get_top_selling_cases()
            return ChatResponse(reply=format_top_selling(top_selling_cases))
        elif static_intent == "how_to_order_faq":
            return ChatResponse(reply=FAQS["how to order"])
        elif static_intent == "my_orders_request":
            if request.user_id:
                orders_html = await get_user_orders(request.user_id)
                return ChatResponse(reply=orders_html)
            else:
                return ChatResponse(reply="Please log in to view your orders.")
        elif static_intent == "faq":
            faq_answer = match_faq(request.message)
            if faq_answer:
                return ChatResponse(reply=faq_answer)

        # Otherwise, use LLM pipeline for product search or general chat
        response = main_pipeline.invoke({"message": request.message})
        return ChatResponse(reply=response)
    except Exception as e:
        print(f"Error during pipeline invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("code:app", host="0.0.0.0", port=8000, reload=True) # Changed from "code:check" to "code:app"