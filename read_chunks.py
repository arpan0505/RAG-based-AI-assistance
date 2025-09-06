import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
    "model":"bge-m3", 
    "input":text_list
})
    
    embeddings = r.json()['embeddings']
    return embeddings

jsons = os.listdir("jsons")

my_dict = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Create embeddings for {json_file}")    
    embeddings = create_embeddings([c['text'] for c in content['chunks']])
    
    for i,chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dict.append(chunk)
        if(i == 5):  #read 5 chunks only for testing
            break
    break

# print(my_dict)
        
df = pd.DataFrame.from_records(my_dict)
print(df)

incoming_query = input("Enter your question: ")
question_embedding = create_embeddings([incoming_query])[0]
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_result = 3
max_index = similarities.argsort()[::-1][0:top_result]
print(max_index)
new_df = df.loc[max_index.flatten()]
print(new_df[["title","number","text"]])
