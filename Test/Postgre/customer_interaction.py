#import mysql.connector 
import psycopg2
import streamlit as st

user = "olasinior"
email = "olasinior@me.com"

connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )
cursor = connection.cursor()

# Step 1: Retrieve the current disliked articles for the user
select_query = "SELECT liked_articles, disliked_articles FROM customers WHERE email = %s;"
cursor.execute(select_query, (email,))
result = cursor.fetchone()

liked, disliked = result[0], result[1]



#connection = mysql.connector.connect(**db_credentials)
interaction = {
    '0': {'upvotes': 1,'downvotes': 0},
    '1': {'upvotes': 0,'downvotes': 1},
    '2': {'upvotes': 1,'downvotes': 0}
}

liked_articles = " ".join([key for key, value in interaction.items() if value['upvotes'] >= 1])
disliked_articles = " ".join([key for key, value in interaction.items() if value['downvotes'] >= 1])

updated_disliked_articles = liked + liked_articles
updated_disliked_articles_json = disliked + disliked_articles

connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )

# Create a cursor to interact with the database
cursor = connection.cursor()

insert_query = """
INSERT INTO customers (email, liked_articles, disliked_articles)
VALUES (%s, %s, %s)
ON CONFLICT (email)
DO UPDATE SET
    liked_articles = EXCLUDED.liked_articles,
    disliked_articles = EXCLUDED.disliked_articles
"""
cursor.execute(insert_query, (email, liked_articles, disliked_articles))


# Commit the transaction and close the connection
connection.commit()
cursor.close()
connection.close()
