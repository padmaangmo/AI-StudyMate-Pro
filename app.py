import ast
import hashlib
import json
import operator
import re
import time
from io import BytesIO

import chromadb
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title='AI StudyMate Pro', page_icon='🎓', layout='wide')
st.title('🎓 AI StudyMate Pro')
st.caption('Codomax Module 6 | Gemini + RAG + ChromaDB + AI Agent Tools')

MODEL = 'gemini-3.5-flash-lite'
API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent'

@st.cache_resource
def embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def vector_client():
    return chromadb.EphemeralClient()


def split_text(text, size=950, overlap=150):
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]


def read_file(file):
    if file.name.lower().endswith('.pdf'):
        return '\n'.join(page.extract_text() or '' for page in PdfReader(BytesIO(file.getvalue())).pages)
    return file.getvalue().decode('utf-8', errors='replace')


def index_document(file):
    content = read_file(file)
    if not content.strip():
        raise ValueError('No readable text found. Scanned PDFs may need OCR.')
    chunks = split_text(content)
    digest = hashlib.sha256(file.getvalue()).hexdigest()[:20]
    name = 'study_' + digest
    collection = vector_client().get_or_create_collection(name=name)
    if collection.count() == 0:
        embeddings = embedding_model().encode(chunks, normalize_embeddings=True).tolist()
        collection.add(ids=[str(i) for i in range(len(chunks))], documents=chunks,
                       embeddings=embeddings, metadatas=[{'chunk': i + 1} for i in range(len(chunks))])
    return name, len(chunks)


def search_notes(question, collection_name, k=3):
    if not collection_name:
        return {'error': 'Upload a PDF or TXT document first.'}
    collection = vector_client().get_collection(collection_name)
    embeddings = embedding_model().encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=embeddings, n_results=min(k, collection.count()),
                               include=['documents', 'metadatas'])
    matches = []
    for document, meta in zip(results['documents'][0], results['metadatas'][0]):
        matches.append({'chunk': meta['chunk'], 'text': document})
    return {'matches': matches}


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
       ast.Mod: operator.mod, ast.Pow: operator.pow,
       ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculator(expression):
    tree = ast.parse(expression.replace('×', '*').replace('÷', '/').replace('^', '**'), mode='eval')
    def compute(node):
        if isinstance(node, ast.Expression):
            return compute(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](compute(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = compute(node.left), compute(node.right)
            if abs(left) > 1e12 or abs(right) > 1e12 or (isinstance(node.op, ast.Pow) and abs(right) > 12):
                raise ValueError('Expression too large')
            return OPS[type(node.op)](left, right)
        raise ValueError('Only arithmetic expressions are supported')
    value = compute(tree)
    if not isinstance(value, (int, float)) or abs(value) > 1e15:
        raise ValueError('Result too large')
    return {'result': value}


def wikipedia_search(topic):
    headers = {'User-Agent': 'AIStudyMatePro/1.0 (educational capstone)'}
    search = requests.get('https://en.wikipedia.org/w/api.php',
                          params={'action': 'query', 'list': 'search', 'srsearch': topic, 'format': 'json', 'srlimit': 1},
                          headers=headers, timeout=15)
    search.raise_for_status()
    hits = search.json().get('query', {}).get('search', [])
    if not hits:
        return {'error': 'No Wikipedia article found'}
    title = hits[0]['title']
    summary = requests.get('https://en.wikipedia.org/w/api.php',
                           params={'action': 'query', 'prop': 'extracts', 'exintro': 1, 'explaintext': 1,
                                   'titles': title, 'format': 'json'}, headers=headers, timeout=15)
    summary.raise_for_status()
    pages = summary.json().get('query', {}).get('pages', {})
    extract = next(iter(pages.values())).get('extract', '')[:3500]
    from urllib.parse import quote
    return {'title': title, 'summary': extract, 'source': 'https://en.wikipedia.org/wiki/' + quote(title.replace(' ', '_'))}


def tool_declarations():
    return [{'functionDeclarations': [
        {'name': 'calculator', 'description': 'Calculate a mathematical arithmetic expression precisely.',
         'parameters': {'type': 'OBJECT', 'properties': {'expression': {'type': 'STRING'}}, 'required': ['expression']}},
        {'name': 'wikipedia_search', 'description': 'Search Wikipedia for factual or historical information.',
         'parameters': {'type': 'OBJECT', 'properties': {'topic': {'type': 'STRING'}}, 'required': ['topic']}},
        {'name': 'search_notes', 'description': 'Search the uploaded PDF/TXT notes for relevant passages. Use for questions about the uploaded document.',
         'parameters': {'type': 'OBJECT', 'properties': {'question': {'type': 'STRING'}}, 'required': ['question']}}
    ]}]


def gemini(contents, key, use_tools=True):
    payload = {'contents': contents, 'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 1600},
               'systemInstruction': {'parts': [{'text': 'You are AI StudyMate Pro, a friendly educational assistant. Use tools for calculations, Wikipedia facts and uploaded notes. For document questions, base answers on retrieved note passages and cite chunk numbers. If the uploaded notes do not support an answer, say so rather than inventing it. Explain clearly and avoid claiming to have used tools you did not call.'}]}}
    if use_tools:
        payload['tools'] = tool_declarations()
    for attempt in range(3):
        
        response = requests.post(API_URL, headers={'x-goog-api-key': key}, json=payload, timeout=(10, 120))

        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()


def agent(question, key, collection_name, history):
    contents = []
    for item in history[-10:]:
        contents.append({'role': 'model' if item['role'] == 'assistant' else 'user',
                         'parts': [{'text': item['content']}]})
    contents.append({'role': 'user', 'parts': [{'text': question}]})
    trace = []
    for _ in range(5):
        data = gemini(contents, key)
        candidate = data['candidates'][0]
        parts = candidate.get('content', {}).get('parts', [])
        calls = [part['functionCall'] for part in parts if 'functionCall' in part]
        if not calls:
            return '\n'.join(part.get('text', '') for part in parts).strip() or 'No response generated.', trace
        contents.append(candidate['content'])
        responses = []
        for call in calls:
            name, args = call['name'], call.get('args', {})
            try:
                if name == 'calculator':
                    result = calculator(args['expression'])
                elif name == 'wikipedia_search':
                    result = wikipedia_search(args['topic'])
                elif name == 'search_notes':
                    result = search_notes(args['question'], collection_name)
                else:
                    result = {'error': 'Unknown tool'}
            except Exception as exc:
                result = {'error': str(exc)}
            trace.append({'tool': name, 'arguments': args, 'result': result})
            responses.append({'functionResponse': {'name': name, 'response': result}})
        contents.append({'role': 'user', 'parts': responses})
    return 'The agent reached its tool-call limit. Try a more specific question.', trace


with st.sidebar:
    st.header('⚙️ Settings')
    st.write('**AI StudyMate Pro**')
    st.write('LLM: Gemini | RAG: MiniLM + ChromaDB')
    st.info('Upload a document to enable notes search. Calculator and Wikipedia work without an upload.')
    if st.button('Clear conversation'):
        st.session_state.messages = []
        st.rerun()

uploaded = st.file_uploader('📄 Upload study notes (PDF or TXT)', type=['pdf', 'txt'])
if uploaded:
    fingerprint = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if st.session_state.get('file_fingerprint') != fingerprint:
        with st.spinner('Extracting text and building embeddings...'):
            try:
                collection_name, chunk_count = index_document(uploaded)
                st.session_state.collection_name = collection_name
                st.session_state.file_fingerprint = fingerprint
                st.session_state.messages = []
                st.success(f'Document indexed successfully: {chunk_count} chunks')
            except Exception as exc:
                st.error(f'Document processing failed: {exc}')
    else:
        st.success('Document ready for questions')
else:
    st.session_state.collection_name = None
    st.session_state.file_fingerprint = None

key = st.secrets.get('GEMINI_API_KEY', '')
if not key:
    st.warning('Add GEMINI_API_KEY in Streamlit Community Cloud → App settings → Secrets.')

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('trace'):
            with st.expander('🔧 Tool execution trace'):
                st.json(message['trace'])

question = st.chat_input('Ask about your notes, research a topic, or solve a calculation...')
if question:
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        if not key:
            st.error('Please configure GEMINI_API_KEY first.')
        else:
            with st.spinner('Agent is thinking and using tools...'):
                try:
                    answer, trace = agent(question, key, st.session_state.get('collection_name'), st.session_state.messages)
                    st.markdown(answer)
                    if trace:
                        with st.expander('🔧 Tool execution trace'):
                            st.json(trace)
                    st.session_state.messages.extend([
                        {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': answer, 'trace': trace}
                    ])
                except requests.exceptions.HTTPError as exc:
                    st.error(f'Gemini/API error: {exc.response.status_code} — {exc.response.text[:350]}')
                except Exception as exc:
                    st.error(f'Agent error: {exc}')
