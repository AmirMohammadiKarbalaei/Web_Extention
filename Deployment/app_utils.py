
import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import defaultdict
import faiss
import pickle
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import numpy as np
from lxml import etree
import requests
import logging
import nest_asyncio
import asyncio
import aiohttp
from sitemaps_utils import *



# Define your asynchronous function

async def fetch_url(session, url, timeout):
    try:
        async with session.get(url, timeout=timeout) as response:
            return await response.text()
    except Exception as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None

async def request_sentences_from_urls_async(urls, timeout=20):
    articles_dict = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, url in enumerate(urls.Url, start=1):
            if (idx - 1) % 100 == 0:
                logging.info(f"\nProcessing URL {idx - 1}/{len(urls)/100}")

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
                    articles_dict[urls["Title"][idx - 1]] = article
                else:
                    logging.warning("No article content found on the page.")
            except Exception as e:
                logging.error(f"Error extracting article content from {url}: {e}")

    return articles_dict
@st.cache_data
def collect_embed_content(df):
    with st.spinner("Fetching news content"):
        data = df[(df.Topic == "news") | (df.Topic == "sport")]
        bbc_news = asyncio.run(articles(data, timeout=20))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
        article_main_body = list(bbc_news.values())


        
    with st.spinner("Embedding news content"):
        # Initialize progress bar
        progress_text = "Embedding content in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        embeddings = []
        total_articles = len(article_main_body)
        
        for i, data in enumerate(article_main_body):
            # Tokenize with padding
            inputs = tokenizer("".join(data), return_tensors="pt", padding='max_length', truncation=True, max_length=512).to(device)  # Move inputs to GPU
            with torch.no_grad():  # No need to track gradients during inference
                embedding = encoder(**inputs).pooler_output
            embeddings.append(embedding)
            
            # Update progress bar
            percent_complete = int((i + 1) / total_articles * 100)
            my_bar.progress(percent_complete, text=progress_text)
        
        # Convert embeddings tensor to numpy arrays
        embeddings_np = [embedding.cpu().numpy() for embedding in embeddings]

        # Convert embeddings to float32
        embeddings_np = [embedding.astype('float32') for embedding in embeddings_np]
        embeddings_np = np.array(embeddings_np).reshape(len(embeddings_np), 768)
        content_embedding = (list(bbc_news.values()), embeddings_np)
        
        st.success("Content has been collected and embedded")
        my_bar.empty()
    
    return content_embedding
# Function to load embeddings
@st.cache_resource
def load_embeddings():
    file_path = 'content_embedding.pkl'

    with open(file_path, 'rb') as file:
        embeddings = pickle.load(file)

    return embeddings

# Function to initialize interactions
def initialize_interactions():
    return defaultdict(lambda: {'upvotes': 0, 'downvotes': 0})

# Function to track interactions
def track_interaction(interactions, news_id, action):
    if action == 'Upvoted':
        interactions[news_id]['upvotes'] += 1
    elif action == 'Downvoted':
        interactions[news_id]['downvotes'] += 1
    print(f"User interacted with news article {news_id} - {action}")
    print(interactions[news_id])

# def request_sentences_from_urls_app(urls, timeout=20):
#     """
#     Extracts the main article content from a list of URLs using `requests` and `lxml`.
#     Returns a dictionary containing article bodies for each URL.
#     Handles fetch errors gracefully and returns None for failed URLs.
#     """

#     articles_dict = {}

#     for idx, url in enumerate(list(urls.Url), start=1):
#         if (idx-1)%50 ==0:
#             st.write(f"\nProcessing URL {(idx-1)//50}/{len(urls)//50}")

#         try:
#             response = requests.get(url, timeout=timeout)
#             response.raise_for_status()  # Raise exception for HTTP errors
#         except requests.RequestException as e:
#             st.error(f"Failed to fetch the web page: {e}")
#             continue
#         except TimeoutError:
#             st.error(f"Timeout occurred while fetching URL: {url}")
#             continue

#         try:
#             tree = etree.HTML(response.content)
#             article_element = tree.find(".//article")
            
#             if article_element is not None:
#                 outer_html = etree.tostring(article_element, encoding='unicode')
                
#                 article_body = remove_elements(outer_html)
                
#                 article = []
#                 for line in (i for i in article_body.split("\n") if len(i) >= 40):
#                     article.append(line)
#                 articles_dict[urls["Title"][idx-1]] = article
#                 #logging.info("Article has been extracted")
#             else:
#                 st.warning("No article content found on the page. URL: {url}")
#                 continue
#         except Exception as e:
#             st.error(f"Error extracting article content: {e} URL: {url}")
#             continue

#     return articles_dict
# Function to print topic counts
def streamlit_print_topic_counts(data_frame: pd.DataFrame, section_name: str):
    st.subheader(section_name)
    topic_counts = data_frame['Topic'].value_counts()
    st.write(topic_counts)
    st.write(f"Total: {topic_counts.sum()}")

# Main function for Streamlit app
st.cache_resource
async def articles(urls, timeout=20):
    articles = await request_sentences_from_urls_async(urls, timeout)
    return articles