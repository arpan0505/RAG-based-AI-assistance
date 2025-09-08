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

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
    "model":"llama3.1", 
    "prompt": prompt,
    "stream": False
})
    
    response = r.json()
    print(response)
    return response

df = joblib.load("embedding.joblib")


incoming_query = input("Enter your question: ")
question_embedding = create_embeddings([incoming_query])[0]
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_result = 10
max_index = similarities.argsort()[::-1][0:top_result]
# print(max_index)
new_df = df.loc[max_index.flatten()]
# print(new_df[["title","number","text"]])

prompt = f'''I am teaching SQL using a structured SQL course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, and the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

-----------------------------------------------------------------------

"{incoming_query}"
The user asked this question related to the video chunks. You must answer by identifying in which video and at what timestamp the relevant SQL concept is explained. Clearly guide the user to the specific video and time range. 

If the user’s question is unrelated to the SQL course content, politely respond that you can only answer questions related to this SQL course.'''


with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

responce = inference(prompt)["response"]
print(responce)

with open("responce.txt", "w", encoding="utf-8") as f:
    f.write(responce)
# for index, item in new_df.iterrows():
#     print(index, item['title'], item['number']  , item['text'], item['start'], item['end'])
   