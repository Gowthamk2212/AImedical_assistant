import os
import uuid
import time
from typing import TypedDict, Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, Form
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import InMemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.messages import HumanMessage

from tavily import TavilyClient
from google import genai

from dotenv import load_dotenv
load_dotenv()

import json

UPLOAD_DIR = "./uploads"
CHAT_DIR = "./chat_history"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHAT_DIR, exist_ok=True)

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not set")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not set")


# Gemini LLM
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# Embeddings + Chroma
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = Chroma(
    collection_name="medical_kb",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Tavily Web Search
tavily = TavilyClient()

# Gemini File Client
client = genai.Client()

# ======================
# STATE
# ======================
class GraphState(TypedDict, total=False):
    query: str
    chat_history: List[Dict]
    rag_answer: Optional[str]
    final_answer: Optional[str]
    quality: Optional[str]

# ======================
# RAG NODE
# ======================
def rag_node(state: GraphState):
    query = state["query"]
    # history = state.get("chat_history", [])
    history = list(state.get("chat_history", []))

    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in history]
    )

    prompt = f"""
You are a medical assistant.

Conversation so far:
{history_text}

Use ONLY the below context:
{context}

Question: {query}
"""

    response = model.invoke(prompt).content

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response})

    MAX_HISTORY = 10
    history = history[-MAX_HISTORY:]
    return {
        "rag_answer": response,
        "chat_history": history
    }

# ======================
# QUALITY EVALUATION NODE
# ======================
def evaluate_node(state: GraphState):
    prompt = f"""
Evaluate the quality of the answer.

Question: {state['query']}
Answer: {state['rag_answer']}

Criteria:
- Is it medically accurate?
- Is it complete?
- Is it relevant?

Return ONLY one word:
GOOD or BAD
"""
    result = model.invoke(prompt).content.strip()
    return {"quality": result}

# ======================
# WEB SEARCH NODE
# ======================

def websearch_node(state: GraphState):
    original_query = state["query"]

    # ======================
    # STEP 1: SUMMARIZE QUERY
    # ======================
    summary_prompt = f"""
Convert the following input into a short web search query.

Rules:
- Max 15 words
- Focus on key medical terms only
- Remove unnecessary details

Input:
{original_query}

Output:
"""

    short_query = model.invoke(summary_prompt).content.strip()

    # HARD SAFETY (Tavily limit)
    short_query = short_query[:350]

    # ======================
    # STEP 2: WEB SEARCH
    # ======================
    try:
        results = tavily.search(query=short_query, max_results=5)
        web_context = "\n".join([r["content"] for r in results["results"]])
        web_context = web_context[:3000]
    except Exception as e:
        web_context = "No web results available."

    # ======================
    # STEP 3: COMBINE
    # ======================
    prompt = f"""
You are a medical assistant.

Question:
{original_query}

RAG Answer:
{state['rag_answer']}

Web Results:
{web_context}

Provide a corrected, concise, and medically accurate final answer.
"""

    final = model.invoke(prompt).content

    # history = state["chat_history"]
    history = list(state.get("chat_history", []))
    history.append({"role": "assistant", "content": final})

    return {
        "final_answer": final,
        "chat_history": history
    }

# ======================
# FINAL NODE (GOOD CASE)
# ======================
def final_node(state: GraphState):
    return {"final_answer": state["rag_answer"]}

# ======================
# ROUTER
# ======================
def route(state: GraphState):
    if state.get("quality", "").strip().upper() == "GOOD":
        return "good"
    return "bad"

# ======================
# BUILD GRAPH
# ======================
builder = StateGraph(GraphState)

builder.add_node("rag", rag_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("websearch", websearch_node)
builder.add_node("final", final_node)

builder.set_entry_point("rag")

builder.add_edge("rag", "evaluate")

builder.add_conditional_edges(
    "evaluate",
    route,
    {
        "good": "final",
        "bad": "websearch"
    }
)

builder.add_edge("websearch", END)
builder.add_edge("final", END)

# memory = InMemorySaver()

# graph = builder.compile(checkpointer=memory)

def load_chat(thread_id):
    path = f"{CHAT_DIR}/{thread_id}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_chat(thread_id, history):
    path = f"{CHAT_DIR}/{thread_id}.json"
    with open(path, "w") as f:
        json.dump(history, f)


graph = builder.compile()

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import uuid
import time

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "running"}

# ======================
# REQUEST MODELS
# ======================
class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    thread_id: str
    response: str

# ======================
# CHAT (JSON INPUT)
# ======================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())

    chat_history = load_chat(thread_id)

    result = graph.invoke(
        {
            "query": req.query,
            "chat_history": chat_history
        }
    )

    updated_history = result.get("chat_history", chat_history)
    save_chat(thread_id, updated_history)

    return {
        "thread_id": thread_id,
        "response": result.get("final_answer", "Error generating response")
    }

import asyncio
# ======================
# FILE UPLOAD (FORM-DATA)
# ======================
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None)
):
    thread_id = thread_id or str(uuid.uuid4())

    # Save file
    # file_path = f"/tmp/{file.filename}"
    file_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Upload to Gemini
    myfile = client.files.upload(file=file_path)

    while myfile.state.name == "PROCESSING":
        # time.sleep(2)
        await asyncio.sleep(2)
        myfile = client.files.get(name=myfile.name)

    # Extract content from PDF
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all medical insights from this report."},
            {
                "type": "file",
                "file_id": myfile.uri,
                "mime_type": "application/pdf",
            },
        ]
    )

    extracted_text = model.invoke([message]).content

    chat_history = load_chat(thread_id)

    # Pass to LangGraph
    result = graph.invoke(
        {
            "query": extracted_text,
            "chat_history": chat_history
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    updated_history = result.get("chat_history", chat_history)
    save_chat(thread_id, updated_history)

    return {
        "thread_id": thread_id,
        # "response": result["final_answer"]
        "response": result.get("final_answer", "Error generating response")
    }