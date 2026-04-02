import os
import sys

# --- RENDER SQLITE VERSION FIX ---
# Necessary for ChromaDB to run on Render's Linux environment
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

# Force CPU to avoid GPU warnings in Render logs
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

app = FastAPI()

# 1. HEALTH CHECK
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

# 2. API INITIALIZATION (Using your new Free Tier Key)
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1'} 
)

try:
    # Ensure 'hits_vectordb' folder is uploaded to your GitHub
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)
except Exception as e:
    print(f"DB Warning: {e}")

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
        
        # Distance validation (1.4 is a safe match threshold)
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        if best_distance > 1.4: 
            return {"response": "I am sorry, I don't have that information. Please contact **info@hindustanuniv.ac.in**."}

        context = "\n".join(results['documents'][0])
        
        # Use a simple prompt structure to avoid 400 errors
        full_prompt = (
            f"You are the HITS Official Assistant. Context: {context}\n\n"
            f"User Question: {clean_query}\n\n"
            f"Answer using ONLY the context. If the answer is missing, give info@hindustanuniv.ac.in."
        )

        # 3. GENERATION CALL
        # 'gemini-1.5-flash' is the most stable choice for Free Tier keys
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=full_prompt
        )
        
        return {"response": response.text}

    except Exception as e:
        print(f"Error: {str(e)}")
        # If 1.5-flash fails, a quick fallback to 1.5-flash-8b (smallest/easiest model)
        try:
             response = client.models.generate_content(
                model="gemini-1.5-flash-8b", 
                contents=full_prompt
            )
             return {"response": response.text}
        except:
            return {"response": "The HITS system is busy. Please contact **info@hindustanuniv.ac.in**."}
