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
            n_results=5, 
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        print(f"DEBUG: Search Distance is {best_distance}")

        # 2. HYBRID LOGIC: Choose the "Vibe" of the response
        if best_distance < 1.7:
            # High/Medium Accuracy - Give the AI the "Expert" Persona here
            persona_prefix = (
                "You are the HITS Expert. Answer based on this context. "
                "Use Markdown tables for data, bold names for alumni, and provide links if seen. "
                "Be detailed and professional.\n\n"
            )
            context = "\n".join(results['documents'][0])
        else:
            # Fallback for very high distances
            return {"response": "I am sorry, I don't have specific data on that. Please contact **info@hindustanuniv.ac.in**."}

        # 3. CONSTRUCT THE FINAL PROMPT (No more 'config' needed)
        full_prompt = f"{persona_prefix}Context: {context}\n\nUser Question: {query.text}"

        # 4. STABLE MODEL LOOP
        model_priority = [
            "gemini-1.5-flash", 
            "gemini-2.5-flash", 
            "gemini-2.0-flash"
        ]

        for model_id in model_priority:
            try:
                print(f"DEBUG: Trying {model_id}...")
                # We simplified this call to avoid the JSON field error
                response = client.models.generate_content(
                    model=model_id,
                    contents=full_prompt
                )
                if response:
                    print(f"DEBUG: Success with {model_id}!")
                    return {"response": response.text}
            except Exception as e:
                print(f"DEBUG: {model_id} failed: {e}")
                continue

        return {"response": "The HITS system is busy. Please contact **info@hindustanuniv.ac.in**."}

    except Exception as final_e:
        print(f"Final Gen Error: {final_e}")
        return {"response": "HITS System Error. Please contact **info@hindustanuniv.ac.in**."}
