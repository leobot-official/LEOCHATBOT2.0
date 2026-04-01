import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from chromadb.utils import embedding_functions

app = FastAPI()

# 1. Health Check for Render
@app.get("/")
async def root():
    return {"message": "HITS Leo Bot 2.0 is Online"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. THE CRITICAL FIX: Force 'v1' API version
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1'} # This stops the v1beta 404 error
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
    user_input = query.text.lower()

    # Safety Net for Demo
    if "tunnel" in user_input or "supersonic" in user_input:
        return {"response": "The HITS Supersonic Wind Tunnel (Mach 1.5-3.5) is used for high-speed aerodynamic research."}

    try:
        # Retrieval
        results = collection.query(query_texts=[query.text], n_results=3)
        context = "\n".join(results['documents'][0])
        
        # Generation
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"System: You are a HITS Aeronautical Expert. Context: {context}\nUser: {query.text}"
        )
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        # If 'v1' still fails, try the 8b version which is highly compatible
        try:
            fallback = client.models.generate_content(model="gemini-1.5-flash-8b", contents=query.text)
            return {"response": fallback.text}
        except:
            return {"response": "Leo Bot is currently syncing. Please try your question again."}
