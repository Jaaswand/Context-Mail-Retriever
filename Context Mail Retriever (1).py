#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


df=pd.read_csv("email dataset.csv")
df


# In[25]:


df.shape


# In[26]:


df.info()


# In[27]:


df.Label.value_counts()


# In[28]:


df.isnull().any()


# In[29]:


import re
import nltk
from nltk.corpus import stopwords


# In[30]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[31]:


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text


# In[32]:


df['text'] = df['Subject'] + " " + df['Body']


# In[33]:


df['cleaned_text'] = df['text'].apply(clean_text)


# In[34]:


print(df['cleaned_text'])


# In[13]:


#pip install transformers pandas


# In[14]:


#import transformers


# In[15]:


#print(transformers.__version__)


# In[35]:


import os
import openai
from scipy.spatial import distance
from sklearn.cluster import KMeans
#import umap
import plotly.express as px


# In[36]:


openai.api_key = " "


# In[37]:


def get_embedding(text_to_embed):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[text_to_embed]
    )
    embedding = response["data"][0]["embedding"]
    return embedding 


# In[38]:


df["embedding"] = df["cleaned_text"].astype(str).apply(get_embedding)
# Reset index
df.reset_index(drop=True, inplace=True)


# In[39]:


print(openai.__version__)


# In[ ]:




