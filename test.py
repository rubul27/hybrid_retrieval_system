import requests
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
import torch
from qdrant_client import QdrantClient, models
import json
# import fitz  # PyMuPDF for PDF extraction
import os
from Utils.configparser import ConfigParser



tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
    return embeddings

def retrieve_from_qdrant(query, top_k=10):
    query_embedding = generate_embeddings([query])[0]
    client = QdrantClient(url="http://localhost:6333")
    search_results = client.search(
        collection_name="{ingestion}",
        query_vector=query_embedding,
        limit=top_k
    )
    return [result.payload for result in search_results]

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="{ingestion}",
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
)

s = retrieve_from_qdrant("Hello")
print(s)