import qdrant_client
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor
import openai
import requests
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
import torch
from qdrant_client import QdrantClient
import json
import fitz  # PyMuPDF for PDF extraction
import os
from Utils.configparser import ConfigParser
from qdrant_client import QdrantClient, models
# from sentence_transformers import SentenceTransformer

# # Correct repo ID
# model_id = 'sentence-transformers/all-MiniLM-L6-v2'

# # Load the model from the Hugging Face Model Hub
# model_embedding = SentenceTransformer(model_id)
from configparser import ConfigParser
class Ingestion:
    def __init__(self):
       # Read configuration file
        # config = ConfigParser()
        # config.read("app.config")
        # embedding_model = config.get("Embedding Model", "embedding_model")
        # Initialize dense embedding model
        embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
    # Function to generate embeddings
    def generate_embeddings(self,texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings

    # Function to extract text from a PDF file
    def extract_text_from_pdf(self,pdf_path):
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
        
    # Function to split text into chunks of specified size
    def split_text_into_chunks(self,text, chunk_size=50):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    # Function to upload data to Qdrant
    def upload_to_qdrant(self,client, vectors, documents):
        client.upload_collection(
                collection_name='{first_collection}',
                vectors=vectors,
                # points=points
                payload=documents
            )
    
    def ingest_data(self):

        # Initialize Qdrant client
        client = QdrantClient(url="http://localhost:6333")

        # Directory containing PDF files
        pdf_directory = "C:/Users/Lenovo/OneDrive/Desktop/MRAG/backend_datascience"


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
                    
                    # Generate embeddings
                    vectors = self.generate_embeddings(chunks)
                    
                    # Prepare documents with file name
                    documents = [{'text': chunk, 'source': pdf_file} for chunk in chunks]
                    
                    # Upload to Qdrant
                    self.upload_to_qdrant(client, vectors, documents)
                    print(f"Content from {pdf_path} uploaded successfully.")
                else:
                    print(f"Failed to extract text from {pdf_path}.")

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="{first_collection}",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
if __name__ == "__main__":
    ingestion = Ingestion()
    ingestion.ingest_data()