import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Load the embedding model (runs locally, downloads once then caches)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Read and split our document into chunks (one line = one chunk)
#building list in one line istead of eriting 5,6 lines code
with open("personalchoices.txt", "r") as file:
    lines = [line.strip() for line in file if line.strip()]

print(f"Loaded {len(lines)} chunks from personalchoices.txt")

# Step 3: Convert each chunk into an (numerical)embedding - 
#.enocde will return numpy array we have to convert it to list
#doing this for chormlab vector database we have to give it simple pythin list
embeddings = embedder.encode(lines).tolist()

# Step 4: Set up ChromaDB and store everything
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="notes")

collection.add(
    documents=lines,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(lines))]
)

print("Done! Your personalchoices are now stored as embeddings in ./chroma_db")