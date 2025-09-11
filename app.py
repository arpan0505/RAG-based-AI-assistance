from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
import os

app = Flask(__name__)

# Load the embeddings and the DataFrame
df = joblib.load("embedding.joblib")

def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embeddings = r.json()['embeddings']
    return embeddings

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False
    })
    response = r.json()
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    incoming_query = data['question']

    question_embedding = create_embeddings([incoming_query])[0]
    similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
    
    top_result = 20
    max_index = similarities.argsort()[::-1][0:top_result]
    new_df = df.loc[max_index.flatten()]

    prompt = f'''I am teaching SQL using a structured SQL course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, and the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

-----------------------------------------------------------------------

"{incoming_query}"
The user asked this question related to the video chunks. You must answer by identifying in which video and at what timestamp the relevant SQL concept is explained. Clearly guide the user to the specific video and time range. 

If the user’s question is unrelated to the SQL course content, politely respond that you can only answer questions related to this SQL course.'''

    response = inference(prompt)["response"]
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
