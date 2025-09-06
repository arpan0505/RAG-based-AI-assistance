import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
    "model":"bge-m3", 
    "input":text_list
})
    
    embeddings = r.json()['embeddings']
    return embeddings

df = joblib.load("embedding.joblib")


incoming_query = input("Enter your question: ")
question_embedding = create_embeddings([incoming_query])[0]
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_result = 5
max_index = similarities.argsort()[::-1][0:top_result]
print(max_index)
new_df = df.loc[max_index.flatten()]
print(new_df[["title","number","text"]])