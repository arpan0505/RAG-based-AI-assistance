import requests
import os
import json
import pandas as pd

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
    

# print(my_dict)
        
df = pd.DataFrame.from_records(my_dict)
print(df)






# a = create_embeddings("My name is Arpan Neog.")
# print(a)