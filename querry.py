import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# Step 1: Load the same embedding model used during ingest
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Connect to our existing ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="notes")

# Step 3: Set up the Groq client for generation
groq_client = Groq()

print("Ask questions about your notes! Type 'quit' to exit.\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        break

    # Step 4: Convert the question into an embedding
    question_embedding = embedder.encode([question]).tolist()

    # Step 5: Search ChromaDB for the most similar chunks - coolection.querry is chromadb search function
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_chunks = results["documents"][0]

    print("\n--- Retrieved context ---")
    for chunk in retrieved_chunks:
        print(f"  - {chunk}")
    print("--------------------------\n")

    # Step 6: Build a prompt that includes the retrieved context
    context = "\n".join(retrieved_chunks)
    prompt = f"""Answer the question using only the context below. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""

    # Step 7: Send it to Groq
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    answer = response.choices[0].message.content
    print(f"Answer: {answer}\n")