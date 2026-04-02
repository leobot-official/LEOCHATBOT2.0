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

# Force CPU for stability on Render
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "online", "bot": "HITS Leo Bot 2.0"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INITIALIZATION (Try v1 first, as it's the 2026 stable standard)
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1'} 
)

try:
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)
except Exception as e:
    print(f"DB Connection Warning: {e}")

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        clean_query = query.text.lower()

        # SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=3,
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        if best_distance > 1.4: 
            return {"response": "I am sorry, I don't have that information. Please contact **info@hindustanuniv.ac.in**."}

        context = "\n".join(results['documents'][0])
        
        full_prompt = (
            f"You are the HITS Official Assistant. Answer based ONLY on the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {clean_query}\n\n"
            f"If the answer is not in the context, say you don't know and provide info@hindustanuniv.ac.in."
        )

        # --- THE MULTI-MODEL FALLBACK LOOP ---
        # This list covers every possible name Google uses in 2026.
        # It will try them one by one until one gives a 200 OK.
        model_names = [
            "gemini-1.5-flash", 
            "gemini-flash-latest", 
            "gemini-2.0-flash", 
            "gemini-3.1-flash",
            "gemini-1.5-flash-8b"
        ]

        response = None
        last_error = ""

        for model_name in model_names:
            try:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=full_prompt
                )
                if response:
                    break # Success! Exit the loop.
            except Exception as e:
                last_error = str(e)
                print(f"Model {model_name} failed: {last_error}")
                continue # Try the next model name

        if response:
            return {"response": response.text}
        else:
            return {"response": f"Model Error. Please verify API Key or Region. Details: {last_error[:50]}"}

    except Exception as e:
        print(f"Critical Error: {str(e)}")
        return {"response": "The HITS system is busy. Please contact **info@hindustanuniv.ac.in**."}
