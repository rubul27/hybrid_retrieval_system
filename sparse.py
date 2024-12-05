import os
import fitz  # PyMuPDF for PDF extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient, models
import numpy as np

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=50):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return None

# Function to generate TF-IDF embeddings
def generate_tfidf_embeddings(texts):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_vectorizer, tfidf_matrix

# Function to upload documents to Qdrant
def upload_to_qdrant(client, vectors, documents):
    client.upload_collection(
        collection_name='{sparse_collection}',
        vectors=vectors.toarray(),
        payload=documents
    )

# Function to perform sparse search
def sparse_search(query, vectorizer, tfidf_matrix, documents, top_k=5):
    query_vector = vectorizer.transform([query])
    cosine_similarities = np.dot(query_vector, tfidf_matrix.T).toarray()[0]
    top_indices = np.argsort(cosine_similarities)[::-1][:top_k]
    results = [(documents[i], cosine_similarities[i]) for i in top_indices]
    return results

# Directory containing PDFs
pdf_directory = "C:/Users/Lenovo/OneDrive/Desktop/MRAG/backend_datascience"

# Extract texts and create chunks
all_chunks = []
documents = []


for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processing {pdf_path}...")

        text = extract_text_from_pdf(pdf_path)
        if text:
            chunks = split_text_into_chunks(text, chunk_size=50)
            all_chunks.extend(chunks)
            documents.extend([{'text': chunk, 'source': pdf_file} for chunk in chunks])

# Generate TF-IDF embeddings for all chunks
vectorizer, tfidf_matrix = generate_tfidf_embeddings(all_chunks)

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="{sparse_collection}",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
# Upload chunks and embeddings to Qdrant
upload_to_qdrant(client, tfidf_matrix, documents)

# Perform a sparse search
query = "Which major subsidiaries does Alphabet own?"
search_results = sparse_search(query, vectorizer, tfidf_matrix, documents, top_k=5)

# Print search results
for result in search_results:
    print(f"Document: {result[0]['source']}, Score: {result[1]}, Text: {result[0]['text']}")
