import requests

def create_embeddings(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
    "model":"bge-m3", 
    "prompt":text
})
    
    embedding = r.json()['embedding']
    return embedding

a = create_embeddings("My name is Arpan Neog.")
print(a)