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

class webingest:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(ConfigParser.read_config("Embedding Model","embedding_model"))
        self.model = AutoModel.from_pretrained(ConfigParser.read_config("Embedding Model","embedding_model"))

    # Function to generate embeddings
    def generate_embeddings(self,texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings
    
    # Function to fetch and parse web content from a URL
    def fetch_and_parse_content(self,url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Example: Extracting structured content
                paragraphs = [p.get_text() for p in soup.find_all('p')]  # Extract all paragraphs
                headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]  # Extract headings
                
                # Combine paragraphs and headings into structured content
                structured_content = {
                    'paragraphs': paragraphs,
                    'headings': headings,
                    'url': url  # Include URL for metadata
                }
                
                return structured_content
            else:
                print(f"Failed to fetch URL: {url}. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching URL: {url}. Exception: {e}")
            return None
        
    # Function to retrieve data from the internet
    def retrieve_internet_data(self,query):
        params = {
        'q': query,
        'api_key': ConfigParser.read_config("SERP API","serp_api_key"),
        'sort': 'date'  # Sort results by date to ensure the latest data is retrieved
        }
        # Prepare the API request
        url = 'https://serpapi.com/search'
        response = requests.get(url, params=params)
        if response.status_code == 200:
            try:
                search_results = response.json()
                search_items = search_results.get('organic_results', [])

                # Extract URLs from search results
                urls = [item.get('link', '') for item in search_items]

                # Initialize Qdrant client
                client = QdrantClient(ConfigParser.read_config("QDrant Connection","url"))

                # Iterate through URLs and fetch structured content
                for url in urls:
                    structured_content = self.fetch_and_parse_content(url)
                    if structured_content:
                        # Example: Prepare vectors (if needed)
                        vectors = [...]  # Prepare vectors (embeddings) if necessary
                        
                        # Prepare metadata or documents
                        documents = [{
                            'url': structured_content['url'],
                            'paragraphs': structured_content['paragraphs'],
                            'headings': structured_content['headings']
                        }]
                        
                        print(documents)
                        texts = [doc['paragraphs'][0] for doc in documents]
                        vectors = self.generate_embeddings(texts)

                        # Upload vectors and documents to Qdrant
                        client.upload_collection(
                            collection_name='web_content',
                            vectors=vectors,
                            payload=documents
                        )
                        print(f"Content from {url} uploaded successfully to Qdrant.")           
                    else:
                        print(f"Failed to fetch or parse content from {url}. Skipping upload.")
            except IndexError as err:
                print(err)
        else:
            print(f"Error fetching search results: {response.status_code} - {response.text}")
        
    # Function to split text into chunks of specified size
    # def split_text_into_chunks(self,text, chunk_size=50):
    #     words = text.split()
    #     chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    #     return chunks