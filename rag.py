import requests
import numpy as np
from pathlib import Path

OLLAMA = 'http://localhost:11434'
EMBD_MODEL = 'nomic-embed-text:latest'
# CHAT_MODEL = 'llama3.1:latest'
CHAT_MODEL = 'mistral-nemo:12b'

def embed(texts):
    r = requests.post(f'{OLLAMA}/api/embeddings', json={'model': EMBD_MODEL, 'input': texts})
    r.raise_for_status()
    return np.array(r.json()['embedding'])

def ask_llm(prompt):
    r = requests.post(f'{OLLAMA}/api/chat', json={
        'model': CHAT_MODEL, 'stream': False,
        'messages':[{'role':'user', 'content':prompt}],
    })
    return r.json()['message']['content']

chunks = []
for file in Path('docs').glob('*.md'):
    chunks += [p.strip() for p in file.read_text(encoding='utf-8').split('\n\n') if p.strip()]

chunk_vectors = embed()
print(f'Indexed {len(chunks)} chunks \n')

def retrieve(question, k=3):
    q = embed([question])[0]
    scores = chunk_vectors @ q / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]