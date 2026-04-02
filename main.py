import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai 
from chromadb.utils import embedding_functions

# Force CPU to keep logs clean
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

# 1. INITIALIZATION: Using the most stable 'v1' endpoint
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

        # SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=3,
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # If no match in DB, stop here and return contact info
        if best_distance > 1.4: 
            return {"response": "I am sorry, I don't have that information. Please contact **info@hindustanuniv.ac.in**."}

        context = "\n".join(results['documents'][0])
        
        # 2. THE PROMPT: Simple and direct
        full_prompt = f"Context: {context}\n\nQuestion: {clean_query}\n\nAnswer only using the context above. If unsure, give info@hindustanuniv.ac.in."

        # 3. THE GENERATION: Using the most stable model string possible
        try:
            # We use 'gemini-1.5-flash' without any prefixes or beta tags
            response = client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=full_prompt
            )
            return {"response": response.text}
        except Exception as e:
            # Absolute final backup if the first attempt fails
            print(f"Primary failed: {e}")
            response = client.models.generate_content(
                model="gemini-1.5-flash-latest", 
                contents=full_prompt
            )
            return {"response": response.text}

    except Exception as e:
        print(f"Final Catch Error: {str(e)}")
        return {"response": "System syncing. Please contact **info@hindustanuniv.ac.in**."}
