from flask import Flask, request, jsonify
from ingest import Ingestion
from websearch import webingest
from RAG import RetrivalEngine

app = Flask(__name__)

@app.route('/ingest', methods=['POST'])
def ingest():
    ingestion = Ingestion()
    ingestion.ingest_data()
    return jsonify({'status': 'success', 'message': 'Data ingested successfully'})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    web_search = data.get('web_search', False)
    
    if web_search:
        webingestfunc = webingest()
        webingestfunc.retrieve_internet_data(query_text)
    
    retrive = RetrivalEngine()
    response = retrive.process_query(query_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
