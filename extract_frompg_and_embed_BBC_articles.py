import psycopg2
from datetime import date
from sentence_transformers import SentenceTransformer
import datetime
import pandas as pd


# Load SBERT model
model_name = 'distilbert-base-nli-stsb-mean-tokens'

model = SentenceTransformer(model_name)

db_credentials = {
    'dbname': 'Web_Extention',
    'user': 'postgres',
    'password': '2080',
    'host': 'localhost',
    'port': '5432'
}

connection = psycopg2.connect(**db_credentials)

# Create a cursor to interact with the database
cursor = connection.cursor()

# SELECT query to retrieve data from the 'bbc_embeddings' table
today = datetime.date.today()

select_query_todays_articles = f"SELECT title,topic,article_content FROM bbc_daily_links WHERE DATE(last_modified) = '{today}';"

# Execute the query
cursor.execute(select_query_todays_articles)

# Fetch all the rows from the result set
rows = cursor.fetchall()

# Create a DataFrame from the fetched data
columns = [desc[0] for desc in cursor.description]  # Get column names
data_frame_from_table = pd.DataFrame(rows, columns=columns)
data_frame_from_table["Title_some_content"] = data_frame_from_table.apply(lambda row: row["title"] + row["article_content"][100:500], axis=1)

embeddings = model.encode(data_frame_from_table["Title_some_content"])
data_frame_from_table["Embedding"] = list(embeddings)
data_frame_from_table['Embedding'] = data_frame_from_table['Embedding'].apply(lambda x: [float(val) for val in x])


connection = psycopg2.connect(**db_credentials)

# Create a cursor to interact with the database
cursor = connection.cursor()
for _, row in data_frame_from_table.iterrows():
    insert_query = "INSERT INTO bbc_embeddings (title, topic, embedding) VALUES (%s, %s, %s);"
    cursor.execute(insert_query, ( row["title"], row["topic"], row["Embedding"]))

connection.commit()    
connection.close()