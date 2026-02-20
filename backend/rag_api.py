from fastapi import FastAPI
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load DB once (FAST)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=embeddings
)

llm = OllamaLLM(model="llama3")


class Query(BaseModel):
    message: str


@app.post("/chat")
def chat(query: Query):

    docs = vectorstore.similarity_search(query.message)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer based ONLY on context:

{context}

Question: {query.message}
"""

    response = llm.invoke(prompt)

    return {"answer": response}
