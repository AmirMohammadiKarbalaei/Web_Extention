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
from app_utils import *

nest_asyncio.apply()
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
