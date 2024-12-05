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

class RetrivalEngine:
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
    
    # Function to retrieve relevant documents from Qdrant
    def retrieve_from_qdrant(self,query, top_k=10):
        query_embedding = self.generate_embeddings([query])[0]
        search_results = qdrant_client.search(
            collection_name="web_content",
            query_vector=query_embedding,
            limit=top_k
        )
        return [result.payload for result in search_results]

    # Function to rank documents based on relevance
    def rank_documents(self,documents, query):
        query_embedding = self.generate_embeddings([query])[0]
        doc_embeddings = [self.generate_embeddings([doc['paragraphs'][0]])[0] for doc in documents if 'paragraphs' in doc and doc['paragraphs']]
        
        if not doc_embeddings:
            print("No document embeddings generated.")
            return []

        similarities = util.pytorch_cos_sim(torch.tensor([query_embedding]), torch.tensor(doc_embeddings))
        ranked_documents = [doc for _, doc in sorted(zip(similarities[0], documents), key=lambda pair: pair[0], reverse=True)]
        return ranked_documents

    # Multi-Head Retrieval Function
    def multi_head_retrieve(self,query, num_heads=3, top_k=10):
        def retrieve(head_index):
            head_query = f"{query} head-{head_index}"
            return self.retrieve_from_qdrant(head_query, top_k)

        with ThreadPoolExecutor(max_workers=num_heads) as executor:
            results = list(executor.map(retrieve, range(num_heads)))

        # Flatten and remove duplicates using JSON strings
        unique_documents = {json.dumps(doc, sort_keys=True) for sublist in results for doc in sublist}
        unique_documents = [json.loads(doc) for doc in unique_documents]
        return results, unique_documents


    # Main function to process the query and generate response
    def process_query(self,query, sub_queries=None, num_heads=3):
        # Step 2: Handle optional sub-queries and multi-head retrieval
        retrieved_documents_heads, retrieved_documents = self.multi_head_retrieve(query, num_heads)

        # Print documents retrieved by each head
        for i, head_docs in enumerate(retrieved_documents_heads):
            print(f"\nDocuments retrieved by head {i + 1}:\n")
            for doc in head_docs:
                print(doc)

        # Step 3: Rank documents based on relevance
        ranked_documents = self.rank_documents(retrieved_documents, query)

        # Print ranked documents
        print("\nRanked documents:\n")
        for doc in ranked_documents:
            print(doc)

        # Step 4: Generate response using OpenAI's GPT-4 model
        context = " ".join([doc['paragraphs'][0] for doc in ranked_documents if 'paragraphs' in doc and doc['paragraphs']])
        # response = openai.Completion.create(
        #     model="gpt-3.5-turbo",
        #     prompt=f"Query: {query}\nContext: {context}",
        #     max_tokens=500
        # )
        from openai import OpenAI
        import os
        openaiclient = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get(ConfigParser.read_config("OPENAI API","api_key")),
        )
        response = openaiclient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Query: {query}\nContext: {context}"}
            ]
        )
        return response.choices[0].text.strip()