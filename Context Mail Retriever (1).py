#!/usr/bin/env python
# coding: utf-8

pip install tiktoken
pip install pgvector
pip install tiktoken psycopg2-binary openai pgvector

import openai
import pandas as pd
import numpy as np
import tiktoken
import psycopg2
import pgvector
import math
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

openai.api_key = " "

df = pd.read_csv('email dataset.csv')
df.head()

#tokenizing
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    if not string:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def get_body_length(body: str) -> int:
    return len(body.split())

def get_embedding_cost(num_tokens: int) -> float:
    return num_tokens / 1000 * 0.0001

def get_total_embeddings_cost(df: pd.DataFrame) -> float:
    total_tokens = sum(num_tokens_from_string(text) for text in df['body'])
    return get_embedding_cost(total_tokens)

# List for chunked content and embeddings
new_list = []


for index, row in df.iterrows():
    text = row['Body']
    token_len = num_tokens_from_string(text)
    
    if token_len <= 512:
        new_list.append([row['Subject'], row['Body'], row['Label'], token_len])
    else:
        # Ideal token size
        ideal_token_size = 512
        ideal_size = int(ideal_token_size // (4/3))  
        words = [word for word in text.split() if word != ' ']  
        total_words = len(words)
        
        
        for start in range(0, total_words, ideal_size):
            end = min(start + ideal_size, total_words)
            chunk = ' '.join(words[start:end])
            chunk_token_len = num_tokens_from_string(chunk)
            
            if chunk_token_len > 0:
                new_list.append([row['Subject'], chunk, row['Label'], chunk_token_len])
                

def get_embedding(text_to_embed):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[text_to_embed]
    )
    embedding = response["data"][0]["embedding"]
    return embedding 
######
for i in range(len(new_list)):
   text = new_list[i][1]
   embedding = get_embeddings(text)
   new_list[i].append(embedding)

# Create a new dataframe from the list
df_new = pd.DataFrame(new_list, columns=['Subject', 'Body', 'Label', 'tokens', 'embeddings'])
df_new.head()

# saving to database
def save_embeddings_to_postgres(df, host, dbname, user, password, table_name):
    # Connect to PostgreSQL database
    conn = psycopg2.connect(host= , dbname= , user= , password= , port= )
    cursor = conn.cursor()


# Create table 
table_create_command = """
CREATE TABLE embeddings (
            id PRIMARY KEY, 
            Subject TEXT,
            Body TEXT,
            Label TEXT,  
            tokens INTEGER,
            embedding VECTOR(1536)
            );
"""
cursor.execute(table_create_command)
cursor.close()
conn.commit()
