#import mysql.connector 
import psycopg2
import pandas as pd
import streamlit as st
import json

import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
#from selenium import webdriver
import pandas as pd
import numpy as np
import datetime
from lxml import etree
import langid
import re
import nltk
import string
import requests
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from bs4 import BeautifulSoup

import asyncio
import aiohttp

logging.basicConfig(level=logging.INFO)



def fetch_and_process_news_data():
    BBC_news_sitemaps = [
        "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
        "https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml",
        "https://www.bbc.com/sitemaps/https-sitemap-com-news-3.xml"
    ]

    sky_news_sitemaps = [
        "https://news.sky.com/sitemap/sitemap-news.xml",
        "https://www.skysports.com/sitemap/sitemap-news.xml"
    ]

    namespaces = {
        'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
        'news': 'http://www.google.com/schemas/sitemap-news/0.9'
    }


    urls = {}
    urls["bbc"] = Extract_todays_urls_from_sitemaps(BBC_news_sitemaps, namespaces, 'sitemap:lastmod')
 
    bbc_topics_to_drop = {"pidgin", "hausa", "swahili", "naidheachdan","cymrufyw"}
    df_BBC = process_news_data(urls, "bbc", bbc_topics_to_drop)
        
    # Uncomment and complete the following if Sky News processing is required
    # sky_topics_to_drop = {"arabic", "urdu"}
    # with st.spinner("Processing Sky News data..."):
    #     df_Sky = process_news_data(urls, "sky", sky_topics_to_drop)
    #     st.write("Sky News data processing complete")
    # return pd.concat([df_BBC, df_Sky])

    return df_BBC.drop_duplicates("Title").reset_index(drop=True)

async def articles(urls, timeout=20):
    articles = await request_sentences_from_urls_async(urls, timeout)
    return articles
def collect_embed_content(df):


    df = df.drop_duplicates(subset="Title").reset_index(drop = True)
    collected_df = asyncio.run(articles(df, timeout=10))

    # st.write("bbc_news:",len(bbc_news.items()),"df:",len(df))

    # st.write(df)
    # for title, content in bbc_news.items():
    #     mask =  df['Title'] == title
    #     df.loc[mask, 'content'] = content
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
    article_main_body = list(collected_df.content)

    # Initialize progress bar

    
    embeddings = []
    total_articles = len(article_main_body)
    
    for i, data in enumerate(article_main_body):
        # Tokenize with padding
        inputs = tokenizer("".join(data), return_tensors="pt", padding='max_length', truncation=True, max_length=512).to(device)  # Move inputs to GPU
        with torch.no_grad():  # No need to track gradients during inference
            embedding = encoder(**inputs).pooler_output
        embeddings.append(embedding)
        
    
    # Convert embeddings tensor to numpy arrays
    embeddings_np = [embedding.cpu().numpy() for embedding in embeddings]
    

    # Convert embeddings to float32
    embeddings_np = [embedding.astype('float32') for embedding in embeddings_np]
    embeddings_np = np.array(embeddings_np).reshape(len(embeddings_np), 768)
    embeddings_list = [embedding for embedding in embeddings_np]
    #content_embedding = (list(bbc_news.values()), embeddings_np)
    #st.write("embeddings_np:",len(embeddings_np),"df:",len(df))

    collected_df["embedding"] = embeddings_list
    

    
    return collected_df

async def fetch_url(session, url, timeout):
    try:
        async with session.get(url, timeout=timeout) as response:
            return await response.text()
    except Exception as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None
async def request_sentences_from_urls_async(urls, timeout=20):
    articles_df = pd.DataFrame(columns=["last_modified","url","topic","title","content"])

    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, url in enumerate(urls.Url, start=1):
            if (idx - 1) % 100 == 0:
                logging.info(f"\nProcessing URL {((idx - 1)//100)+1}/{(len(urls)//100)+1}")

            tasks.append(fetch_url(session, url, timeout))

        results = await asyncio.gather(*tasks)

        for idx, (url, result) in enumerate(zip(urls.Url, results), start=1):
            if result is None:
                continue

            try:
                tree = etree.HTML(result)
                article_element = tree.find(".//article")
                if article_element is not None:
                    outer_html = etree.tostring(article_element, encoding='unicode')
                    article_body = remove_elements(outer_html)
                    article = [line for line in article_body.split("\n") if len(line) >= 40]
                    articles_df.loc[idx - 1] = (urls["Last Modified"][idx - 1],urls["Url"][idx - 1],urls["Topic"][idx - 1],urls["Title"][idx - 1]," ".join(article))
                else:
                    # If no <article> element is found, try using BeautifulSoup with the specific ID
                    soup = BeautifulSoup(result, 'html.parser')
                    article_id = 'main-content'  # Replace with the actual ID you are targeting
                    article_element = soup.find(id=article_id)
                    if article_element:
                        article_body = remove_elements(str(article_element))
                        article = [line for line in article_body.split("\n") if len(line) >= 40]
                        articles_df.loc[idx - 1] = (urls["Last Modified"][idx - 1],urls["Url"][idx - 1],urls["Topic"][idx - 1],urls["Title"][idx - 1]," ".join(article))
                    else:
                        logging.warning(f"No article content found on the page with ID {article_id}.")
            except Exception as e:
                logging.error(f"Error extracting article content from {url}: error: {e}")
    
    articles_df = articles_df[
            (~articles_df['title'].str.contains('weekly round-up', case=False)) & 
            (articles_df['title'] != 'One-minute World News')].drop_duplicates(subset="title").reset_index(drop=True)


    return articles_df
def remove_elements(input_string: str) -> str:
    """
    This function removes all HTML tags and their content from the input string,
    and removes specific patterns such as "Published X hours ago" and "Image source".
    """
    # Parse the input string as HTML
    soup = BeautifulSoup(input_string, 'html.parser')
    
    # Remove <script>, <style>, and <picture> tags and their content
    for script in soup(["script", "style", "picture"]):
        script.decompose()
    
    # Extract text from the parsed HTML
    cleaned_string = soup.get_text(separator='|', strip=True)
    splitted = cleaned_string.split("|")
    splitted_20 = [i for i in splitted if len(i)>20]
    splitted_20 = ". ".join(splitted_20)
    
    
    

    # Define regular expressions to remove unwanted patterns
    patterns = [
        r'\bPublished.*?\bago\b',  # Matches "Published X hours ago"
        r'\bImage source\b',   # Matches "Image source, ..."
        r'\bImage caption\b',  # Matches "Image caption, ..."
        r'\bMedia caption\b',  # Matches "Media caption, ..."
        r'\bGetty Images\b', 
        r'\bBBC Wales News\b',         # Matches "BBC Wales News"
        r'\bPublished\s\d{1,2}\s\w+\b',
        r'\bRelated Internet\b.*', 
        r'\bBBC News Staff\b',
        r'\bFollow\s.*?\snews.*\b',
        r'\b\w+/\w+\b',
        r'Follow\sBBC.*',
        r'["\',]+',
        r'\s+',

 
    ]

    # Remove the matched patterns from the text
    for pattern in patterns:
        splitted_20 = re.sub(pattern, ' ', splitted_20, flags=re.DOTALL)


    stuff_to_drop = ["fully signed version","Use this form to ask your question:","If you are reading this page and can t see the form you will need to visit the mobile version of the","to submit your question or send them via email to",
                 "YourQuestions@bbc.co.uk","Please include your name age and location with any question you send in","Related internet links","This video can not be played","To play this video you need to enable JavaScript in your browser"
                 ,"Sign up for our morning newsletter and get BBC News in your inbox.","Watch now on BBC iPlayer","Listen now on BBC Sounds","Week in pictures"]
    for i in stuff_to_drop:
        splitted_20 = splitted_20.replace(i,"")

    
    cleaned_string = re.sub(r'\.\.|\.\s\.', '.', splitted_20)

    
    article = []
    for line in (i for i in cleaned_string.split("\n") if len(i) >= 10):
        article.append(line)

    return " ".join(article)

def Extract_todays_urls_from_sitemaps(sitemaps: list, namespaces: dict, date_tag: str):
    """
    This function extracts the URLs and Last Modified
    and Title from XML sitemaps for today's date from given sitemaps
    
    Args:
    sitemaps (list): List of sitemap URLs to parse.
    namespaces (dict): Dictionary of XML namespaces.
    date_tag (str): The XML tag used to find the date (e.g., 'sitemap:lastmod' or 'news:publication_date').

    Returns:
    dict: Dictionary of URLs with their last modified date and title.
    """
    sitemap_data = {}
    today = datetime.date.today().isoformat()

    for sitemap in sitemaps:
        response = requests.get(sitemap)
        sitemap_xml = response.content

        root = etree.fromstring(sitemap_xml)

        urls = root.findall('.//sitemap:url', namespaces=namespaces)

        # Extract lastmod, URL, and news title for each <url> element with today's last modified date
        for url in urls:
            lastmod = url.findtext(f'.//{date_tag}', namespaces=namespaces)
            loc = url.findtext('.//sitemap:loc', namespaces=namespaces)
            news_title = url.findtext('.//news:title', namespaces=namespaces)
            
            if lastmod and lastmod.startswith(today):
                sitemap_data[loc] = {
                    'index':loc,
                    'Last Modified': lastmod,
                    'Title': news_title
                }

    return sitemap_data


def clean_text(text):
    """
    The `clean_text` function performs text preprocessing 
    by removing punctuation, stopwords, and applying stemming and 
    lemmatization. It returns the cleaned text for further analysis, 
    making it useful for various natural language processing tasks.
    
    """
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Download stopwords if not already downloaded
    stop_words = set(stopwords.words('english'))

    # Split the text into individual words
    words = text.split()

    # Remove stop words from the text
    words = [word for word in words if word.lower() not in stop_words]

    # Initialize the stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Apply stemming and lemmatization to each word
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]

    return ' '.join(words)


def process_news_data(urls, source_name, topics_to_drop):
    # Create DataFrame from the dictionary
    df_news = pd.DataFrame.from_dict(urls[source_name], orient='index').reset_index(drop=True).rename(columns={'index': 'Url'})

    
    # Remove non-English entries
    df_news = df_news[df_news['Title'].apply(is_english_sentence)]

    df_news = df_news[~df_news['Url'].str.contains('live')] #remove live links
    
    # Extract topics from 'Url'
    df_news['Topic'] = df_news['Url'].str.extract(r'com/([^/]+)/')[0]
    # edge_1 = df_news['Url'].str.extract(r'/([^/]+)/([^/]+)/([^/]+)/([^/]+)/')
    # df_news.drop(index=edge_1[(edge_1[1] == "live") | (edge_1[2] == "live") | (edge_1[3] == "live")].index, inplace=True)

    
    # Drop rows with unwanted topics
    df_news = df_news[~df_news['Topic'].isin(topics_to_drop)].reset_index(drop=True)
    

    
    return df_news

def is_english_sentence(sentence:str):
    """
    This function takes a sentence as input and uses the langid 
    library to classify the language of the sentence and returns True 
    if the sentence is in english.
    """
    lang, confidence = langid.classify(sentence)
    return lang == 'en'


df_BBC = fetch_and_process_news_data()
df_BBC = df_BBC[
            (~df_BBC['Title'].str.contains('weekly round-up', case=False)) & 
            (df_BBC['Title'] != 'One-minute World News')].drop_duplicates(subset="Title").reset_index(drop=True)

df = collect_embed_content(df_BBC)



connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )

# Create a cursor to interact with the database
cursor = connection.cursor()

# Loop through the DataFrame and insert articles if they don't already exist
for _, row in df.iterrows():
    title = row["title"]
    url = row["url"]
    
    # Check if the article already exists
    check_query = "SELECT COUNT(*) FROM articles WHERE title = %s OR url = %s"
    cursor.execute(check_query, (title, url))
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Article doesn't exist, proceed with insertion
        array_list = row["embedding"].tolist()
        array_json = json.dumps(array_list)
        insert_query = """
        INSERT INTO articles (last_modified,title, url, topic, embedding)
        VALUES (%s,%s, %s, %s, %s);
        """
        cursor.execute(insert_query, (row["last_modified"],title, url, row["topic"], array_json))

# Commit the transaction and close the connection
connection.commit()
cursor.close()
connection.close()