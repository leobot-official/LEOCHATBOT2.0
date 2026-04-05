import os
import sys

# --- RENDER SQLITE VERSION FIX ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai 
from chromadb.utils import embedding_functions

os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "online", "bot": "HITS Leo Bot 2.0 Combined"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INITIALIZATION
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1'}
)

try:
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)
except Exception as e:
    print(f"DB Error: {e}")

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        clean_query = query.text.lower()
        
        # 1. SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=5, # Higher results for more context
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        print(f"DEBUG: Search Distance is {best_distance}")

        # 2. DEFINE SYSTEM INSTRUCTIONS BASED ON DISTANCE
        if best_distance < 1.4:
            # High Accuracy Mode (Strict Context)
            sys_instr = "You are the HITS Expert. Use the provided context to answer. Use Markdown tables for specs and bullets for fees. If not found, give info@hindustanuniv.ac.in."
            context = "\n".join(results['documents'][0])
        elif best_distance < 1.8:
            # Helpful Assistant Mode (Loose Context)
            sys_instr = "The user might have a typo or be asking a general question. Use the context as a guide, but be helpful. Suggest corrections if the word looks like 'HITSEEE' or 'EEE'."
            context = "\n".join(results['documents'][0])
        else:
            # Fallback Mode
            return {"response": "I am sorry, I don't have that information in my current database. Please contact **info@hindustanuniv.ac.in**."}

        # 3. MODEL PRIORITY LOOP (The Code 2 Stability)
        model_priority = [
            "gemini-1.5-flash", # Put 1.5 first to avoid the 2.0 quota errors
            "gemini-2.5-flash", 
            "gemini-2.0-flash"
        ]

        full_prompt = f"Context: {context}\n\nUser Question: {query.text}"

        for model_id in model_priority:
            try:
                print(f"DEBUG: Trying {model_id}...")
                response = client.models.generate_content(
                    model=model_id,
                    contents=full_prompt,
                    config={"system_instruction": sys_instr}
                )
                if response:
                    return {"response": response.text}
            except Exception as e:
                print(f"DEBUG: {model_id} failed: {e}")
                continue

        return {"response": "The HITS system is busy. Please contact **info@hindustanuniv.ac.in**."}

    except Exception as final_e:
        print(f"Final Gen Error: {final_e}")
        return {"response": "System syncing. Please contact **info@hindustanuniv.ac.in**."}
