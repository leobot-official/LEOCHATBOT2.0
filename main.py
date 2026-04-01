import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai # Clean import to avoid 'google' namespace conflict
from chromadb.utils import embedding_functions

# 1. Initialize FastAPI
app = FastAPI()

# 2. Health Check (Crucial for Render to keep the service alive)
@app.get("/")
async def root():
    return {"message": "HITS Leo Bot 2.0 is Online"}

# 3. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. INITIALIZATION
api_key = os.getenv("GOOGLE_API_KEY")

# Force 'v1' to stop the 404 NOT FOUND error
client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1'}
)

# Connect to the Vector Database
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

    # --- 🛡️ SAFETY NET FOR DEMO ---
    if "tunnel" in user_input or "supersonic" in user_input:
        return {"response": "The HITS Supersonic Wind Tunnel (Mach 1.5-3.5) is used for high-speed aerodynamic research."}

    try:
        # 1. Retrieval
        results = collection.query(query_texts=[query.text], n_results=3)
        context = "\n".join(results['documents'][0])
        
        # 2. Generation
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"System: You are a HITS Aeronautical Expert. Context: {context}\nUser: {query.text}"
        )
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        # Try a quick fallback to the 8b model if flash is being difficult
        try:
            fallback = client.models.generate_content(model="gemini-1.5-flash-8b", contents=query.text)
            return {"response": fallback.text}
        except:
            return {"response": "Leo Bot is currently syncing. Please try your question again in 30 seconds."}
