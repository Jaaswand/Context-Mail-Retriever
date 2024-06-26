#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("email dataset.csv")
df

df.shape
df.info()

df.Label.value_counts()

df.isnull().any()
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['text'] = df['Subject'] + " " + df['Body']

df['cleaned_text'] = df['text'].apply(clean_text)

print(df['cleaned_text'])

import os
import openai
from scipy.spatial import distance
from sklearn.cluster import KMeans
#import umap
import plotly.express as px

openai.api_key = " "

def get_embedding(text_to_embed):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[text_to_embed]
    )
    embedding = response["data"][0]["embedding"]
    return embedding 


df["embedding"] = df["cleaned_text"].astype(str).apply(get_embedding)
# Reset index
df.reset_index(drop=True, inplace=True)

print(openai.__version__)




