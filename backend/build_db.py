# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings



# 1️⃣ Load document
# loader = TextLoader("knowledge.txt")
print("Loading PDF...")
loader = PyPDFLoader(
    "why-intelligence-fails-lessons-from-the-iranian-revolution-and-the-iraq-war-9780801458859_compress.pdf"
)
documents = loader.load()

# 2️⃣ Split into chunks
print("Splitting text...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3️⃣ Create embeddings
print("Creating embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4️⃣ Store in vector DB
print("Saving to DB...")
Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./db"
)

print("Done.")

# # 4️⃣ Store in vector DB
# vectorstore = Chroma.from_documents(
#     docs,
#     embeddings,
#     persist_directory="./db"
# )

# # 5️⃣ Load LLaMA
# llm = OllamaLLM(model="llama3")

# # 6️⃣ Ask question
# query = input("Ask something: ")

# # 7️⃣ Retrieve relevant chunks
# retrieved_docs = vectorstore.similarity_search(query)

# context = "\n".join([doc.page_content for doc in retrieved_docs])

# # 8️⃣ Send context to LLaMA
# prompt = f"""
# Answer based ONLY on this context:

# {context}

# Question: {query}
# """

# response = llm.invoke(prompt)

# print(response)
