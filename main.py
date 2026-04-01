import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# 1. Load Environment Variables
load_dotenv()

app = FastAPI()

# 2. CORS Middleware (Keep this first!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. GLOBAL INITIALIZATION (Crucial: Define 'client' here)
client = genai.Client() # This is the line your code was missing/couldn't see
db_client = chromadb.PersistentClient(path="./hits_vectordb")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)

class Query(BaseModel):
    text: str


@app.post("/chat")
async def chat(query: Query):
    try:
        # 1. Retrieval: Get the most relevant HITS data
        results = collection.query(query_texts=[query.text], n_results=3) # Increased to 3 for better detail
        context = "\n".join(results['documents'][0])
        
        # 2. Generation: One clear call with the Expert System instructions
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={
                'system_instruction': """
                You are the official HITS Aeronautical Engineering Expert. 
                Use the provided context to answer. If the answer isn't in the context, 
                kindly inform the user you specialize in HITS Aeronautical data.
                
                Key Areas:
                - Specializations: UAV, Satellite Tech, Space Dynamics.
                - Labs: Supersonic Wind Tunnel (Mach 2.0-3.5), ALSIM Flight Simulator, Aircraft Hangars.
                - Alumni: Always mention the PDF link for full lists: 
                https://api.hindustanuniv.ac.in/uploads/Prominent_Alumni_03dd0ed53d.pdf
                - Placements: Mention Boeing, Airbus, ISRO, and HAL.
                
                Tone: Professional, helpful, and technically accurate.
                """
            },
            contents=f"Context: {context}\nQuestion: {query.text}"
        )
        return {"response": response.text}
    except Exception as e:
        print(f"Error: {e}")
        return {"response": "I'm having trouble accessing the HITS database right now."}
    try:
        # Retrieval
        results = collection.query(query_texts=[query.text], n_results=2)
        context = "\n".join(results['documents'][0])
        
        # Generation using the global 'client'
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={'system_instruction': "You are Leo Bot 2.0. Use HITS context only."},
            contents=f"Context: {context}\nQuestion: {query.text}"
        )
        response = client.models.generate_content(
    model="gemini-2.5-flash",
    config={
        'system_instruction': """
        You are the official HITS Aeronautical Engineering Expert. 
        Your knowledge covers:
        1. Specializations (UAV, Satellite Tech, Space Dynamics).
        2. Labs (Supersonic Wind Tunnel, Fatigue & Damage Lab, Simulation Lab).
        3. Alumni: If asked about alumni, list key names and provide the PDF link: https://hindustanuniv.ac.in/pdf/aeronautical_alumni.pdf.
        4. Career: Focus on placements in Boeing, Airbus, and ISRO.
        
        If a user prompt is vague, ask for clarification. Be professional and technical.
        """
    },
    contents=f"Context: {context}\nQuestion: {query.text}"
)
        return {"response": response.text}
    except Exception as e:
        print(f"Error occurred: {e}") # This helps you see errors in the terminal
        return {"response": f"Sorry, I encountered an error: {str(e)}"}