import os
import fitz  # PyMuPDF for PDF extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
class Ingestion:
    def __init__(self, pdf_directory, qdrant_url):
        self.pdf_directory = pdf_directory
        self.client = QdrantClient(url=qdrant_url)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.dense_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def split_text_into_chunks(self, text, chunk_size=50):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def extract_text_from_pdf(self, pdf_path):
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

    def generate_dense_embeddings(self, texts):
        return self.dense_model.encode(texts)

    def generate_tfidf_embeddings(self, texts):
        return self.tfidf_vectorizer.fit_transform(texts).toarray()

    def upload_to_qdrant(self, vectors, documents, collection_name):
        self.client.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=documents
        )

    def ingest_data(self):
        # Initialize Qdrant client
        client = self.client

        # Directory containing PDF files
        pdf_directory = self.pdf_directory

        all_chunks = []
        documents = []

        # Process each PDF file
        for pdf_file in os.listdir(pdf_directory):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, pdf_file)
                print(f"Processing {pdf_path}...")

                # Extract text from PDF
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    # Split text into chunks of 50 words
                    chunks = self.split_text_into_chunks(text, chunk_size=50)
                    all_chunks.extend(chunks)
                    documents.extend([{'text': chunk, 'source': pdf_file} for chunk in chunks])
                else:
                    print(f"Failed to extract text from {pdf_path}.")

        # Generate dense embeddings for all chunks
        dense_vectors = self.generate_dense_embeddings(all_chunks)
        # Generate TF-IDF embeddings for all chunks
        tfidf_vectors = self.generate_tfidf_embeddings(all_chunks)

        # Upload dense embeddings to Qdrant
        self.upload_to_qdrant(dense_vectors, documents, '{dense_collection}')
        print("Dense embeddings uploaded successfully.")

        # Upload TF-IDF embeddings to Qdrant
        self.upload_to_qdrant(tfidf_vectors, documents, 'sparse_collection')
        print("TF-IDF embeddings uploaded successfully.")

# Example usage
pdf_directory = "C:/Users/Lenovo/OneDrive/Desktop/MRAG/backend_datascience"
qdrant_url = "http://localhost:6333"
client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="{dense_collection}",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
client.create_collection(
    collection_name="sparse_collection",
    vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
)
ingestor = Ingestion(pdf_directory, qdrant_url)
ingestor.ingest_data()




