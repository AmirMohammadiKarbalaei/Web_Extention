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


from sitemaps_utils import Extract_todays_urls_from_sitemaps, process_news_data,remove_elements

# Function to fetch and process news data
@st.cache_data
def fetch_and_process_news_data():
    with st.spinner("Fetching and processing todays news ..."):
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

        with st.spinner("Extracting URLs from BBC sitemaps..."):
            urls = {}
            urls["bbc"] = Extract_todays_urls_from_sitemaps(BBC_news_sitemaps, namespaces, 'sitemap:lastmod')
            st.write(f"Found {len(urls['bbc'])} URLs from BBC")

        with st.spinner("Extracting URLs from Sky News sitemaps..."):
            urls["sky"] = Extract_todays_urls_from_sitemaps(sky_news_sitemaps, namespaces, 'news:publication_date')
            st.write(f"Found {len(urls['sky'])} URLs from Sky News")

        bbc_topics_to_drop = {"pidgin", "hausa", "swahili", "naidheachdan"}
        with st.spinner("Processing BBC news data..."):
            df_BBC = process_news_data(urls, "bbc", bbc_topics_to_drop)
            st.write("BBC news data processing complete")

        # Uncomment and complete the following if Sky News processing is required
        # sky_topics_to_drop = {"arabic", "urdu"}
        # with st.spinner("Processing Sky News data..."):
        #     df_Sky = process_news_data(urls, "sky", sky_topics_to_drop)
        #     st.write("Sky News data processing complete")
        # return pd.concat([df_BBC, df_Sky])

    return df_BBC

def collect_embed_content(df):
    with st.spinner("Fetching news content"):
        data = df[(df.Topic =="news") | (df.Topic == "sport")]
        bbc_news = request_sentences_from_urls_app(data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
        article_main_body = list(bbc_news.values())
        embeddings = []
        with st.spinner("Embedding content"):
        # Process and encode each article body
            for i, data in enumerate(article_main_body):
                # Tokenize with padding
                inputs = tokenizer("".join(data), return_tensors="pt", padding='max_length', truncation=True, max_length=512).to(device)  # Move inputs to GPU
                with torch.no_grad():  # No need to track gradients during inference
                    embedding = encoder(**inputs).pooler_output
                embeddings.append(embedding)
                if i % 10 == 0:
                    print(f"{i // 10}/{len(article_main_body) // 10}")
            # Convert embeddings tensor to numpy arrays
            embeddings_np = [embedding.cpu().numpy() for embedding in embeddings]

            # Convert embeddings to float32
            embeddings_np = [embedding.astype('float32') for embedding in embeddings_np]
            embeddings_np = np.array(embeddings_np).reshape(len(embeddings_np), 768)
            content_embedding = (list(bbc_news.values()),embeddings_np)
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

def request_sentences_from_urls_app(urls, timeout=20):
    """
    Extracts the main article content from a list of URLs using `requests` and `lxml`.
    Returns a dictionary containing article bodies for each URL.
    Handles fetch errors gracefully and returns None for failed URLs.
    """

    articles_dict = {}

    for idx, url in enumerate(list(urls.Url), start=1):
        if (idx-1)%50 ==0:
            st.write(f"\nProcessing URL {idx-1//50}/{len(urls)//50}")

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise exception for HTTP errors
        except requests.RequestException as e:
            st.error(f"Failed to fetch the web page: {e}")
            continue
        except TimeoutError:
            st.error(f"Timeout occurred while fetching URL: {url}")
            continue

        try:
            tree = etree.HTML(response.content)
            article_element = tree.find(".//article")
            
            if article_element is not None:
                outer_html = etree.tostring(article_element, encoding='unicode')
                
                article_body = remove_elements(outer_html)
                
                article = []
                for line in (i for i in article_body.split("\n") if len(i) >= 40):
                    article.append(line)
                articles_dict[urls["Title"][idx-1]] = article
                #logging.info("Article has been extracted")
            else:
                st.warning("No article content found on the page. URL: {url}")
                continue
        except Exception as e:
            st.error(f"Error extracting article content: {e} URL: {url}")
            continue

    return articles_dict
# Function to print topic counts
def streamlit_print_topic_counts(data_frame: pd.DataFrame, section_name: str):
    st.subheader(section_name)
    topic_counts = data_frame['Topic'].value_counts()
    st.write(topic_counts)
    st.write(f"Total: {topic_counts.sum()}")

# Main function for Streamlit app
def main():
    st.title('News App')

    df_news = fetch_and_process_news_data()
    interactions = initialize_interactions()

    streamlit_print_topic_counts(df_news, 'Today\'s Topic Distribution')

    topics = df_news['Topic'].unique()
    selected_topic = st.selectbox('Select a topic:', topics)

    selected_news = df_news[df_news['Topic'] == selected_topic]
    content_embedding = collect_embed_content(df_news)

    st.subheader(f'Latest {selected_topic} News')
    for index, row in selected_news.iterrows():
        st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                <h4>{row['Title']}</h4>
                <p><a href="{row['Url']}" target="_blank">Read more</a></p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Upvote", key=f"upvote_{index}"):
                track_interaction(interactions, index, 'Upvoted')
        with col2:
            if st.button("👎 Downvote", key=f"downvote_{index}"):
                track_interaction(interactions, index, 'Downvoted')

    st.sidebar.subheader('Suggestions')

    most_upvoted = sorted(interactions.items(), key=lambda x: x[1]['upvotes'], reverse=True)

    
    
    embeddings = content_embedding[1]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    for news_id, counts in most_upvoted:
        if counts['upvotes'] > 0:
            k = 3
            D, I = index.search(embeddings, k=k)
            for i in range(1,k,1):
                    news_row = df_news.iloc[I[news_id][i]]
                    st.sidebar.write(f"{news_row['Title']}")
                    st.sidebar.write(f"Source: {news_row['Url']}")

if __name__ == '__main__':
    main()
