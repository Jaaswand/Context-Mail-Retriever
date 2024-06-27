# Context-Mail-Retriever

The Context-Based Email Retrieval System is designed to help users retrieve relevant emails based on a given query. The system provides an efficient way to search through a dataset of emails and return the most relevant ones.

**Key Features:**

**Email Preprocessing:** Clean and preprocess email content. It includes tokenizing, handling long texts (If the email body exceeds 512 tokens, it is split into smaller chunks.)
<br/>**Embedding Generation:** Generate embeddings for email content using OpenAI API.
<br/>**Creating a New DataFrame:** A new DataFrame df_new is created from new_list, containing the subject, body, label, token count, and embeddings for each text.
<br/>**Embedding Storage:** use PostgreSQL as a vector database and store OpenAI embedding vectors (along with original text) using pgvector.
<br/>**Context-Based Retrieval:** Retrieve relevant emails based on user queries using cosine similarity.


# Connecting Python to PostgreSQL
<br/>conn = psycopg2.connect(database = " ", 
                        user = " ", 
                        host= " ",
                        password = " ",
                        port =  )
