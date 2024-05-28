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
            #st.write(f"Found {len(urls['bbc'])} articles from BBC")

        # with st.spinner("Extracting URLs from Sky News sitemaps..."):
        #     urls["sky"] = Extract_todays_urls_from_sitemaps(sky_news_sitemaps, namespaces, 'news:publication_date')
        #     st.write(f"Found {len(urls['sky'])} URLs from Sky News")

        bbc_topics_to_drop = {"pidgin", "hausa", "swahili", "naidheachdan","cymrufyw"}
        with st.spinner("Processing BBC news data..."):
            df_BBC = process_news_data(urls, "bbc", bbc_topics_to_drop)
            
            #st.write("Data processing complete!")

        # Uncomment and complete the following if Sky News processing is required
        # sky_topics_to_drop = {"arabic", "urdu"}
        # with st.spinner("Processing Sky News data..."):
        #     df_Sky = process_news_data(urls, "sky", sky_topics_to_drop)
        #     st.write("Sky News data processing complete")
        # return pd.concat([df_BBC, df_Sky])

    return df_BBC.drop_duplicates("Title").reset_index(drop=True)
def main():
    
    st.title('News App')
    
    with st.status("Collecting data...", expanded=True) as status:
        df_BBC = fetch_and_process_news_data()
        df_BBC = df_BBC[
            (~df_BBC['Title'].str.contains('weekly round-up', case=False)) & 
            (df_BBC['Title'] != 'One-minute World News')].drop_duplicates(subset="Title").reset_index(drop=True)    
        st.write(f"Fetched & Processed {len(df_BBC)} articles from BBC.")
        status.update(label=f"Fetched {len(df_BBC)} articles!")
        interactions = initialize_interactions()
        st.write("embedding data...")

        

        
        df = collect_embed_content(df_BBC)
        status.update(label="Download complete!", state="complete", expanded=False)
    streamlit_print_topic_counts(df_BBC,'Today\'s Topics:')
    preferences = st.multiselect(
    "What are your favorite Topics",
    ["news", "sport", "weather","newsround"])
    selected_topic = st.radio('Select a topic to show:', list(preferences),horizontal =True)
    if len(preferences)>0:
        st.subheader(f'Latest {selected_topic} News')
    else:
        st.subheader(f'Please choose your preferences')

    
    selected_news = df_BBC[df_BBC['Topic'] == selected_topic]
    
    for index, row in selected_news.iterrows():
        st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                <h4>{row['Title']}</h4>
                <p><a href="{row['Url']}" target="_blank">Read more...</a></p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Upvote", key=f"upvote_{index}"):
                track_interaction(interactions, index, 'Upvoted')
        with col2:
            if st.button("👎 Downvote", key=f"downvote_{index}"):
                track_interaction(interactions, index, 'Downvoted')

    
    # LOGO_URL_LARGE = "News-icon.jpg"
    # st.sidebar.image(LOGO_URL_LARGE)

    


    most_upvoted = sorted(interactions.items(), key=lambda x: x[1]['upvotes'], reverse=True)

    sorted_upvoted_idxs = [i[0] for idx, i in enumerate(most_upvoted) if i[1]["upvotes"] > 0]

    embeddings = [np.array(i) for i in df.embedding]
    embeddings_np = np.array(embeddings)


    # index = faiss.IndexFlatL2(np.array(embeddings).shape[1])
    # index.add(np.array(embeddings))
    # k = 5
    # # Perform a search to get the k nearest neighbors
    # D, I = index.search(np.array(embeddings), k=k)


    # related_articles = []
    # for news_idx in sorted_upvoted_idxs:
    #     if df.topic[int(news_idx)] not in preferences:
    #             print("Select Preferences please")
    #             continue
    # related_articles.append(I[int(news_idx)][1:])
    # from collections import Counter
    # flattened_list = [num for sublist in related_articles for num in sublist]

    # # Step 2: Count the occurrences of each number
    # counts = Counter(flattened_list)

    # # Step 3: Sort the numbers based on their counts in descending order
    # sorted_numbers = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # # Step 4: Extract the sorted numbers based on their counts
    # result = [num for num, count in sorted_numbers]

    # for i in result:  # Start from 1 to skip the first neighbor (itself)
    #         # Fetch the news row corresponding to the current neighbor
    #         news_row = df.iloc[i]
            
    #         # Check if the neighbor's topic is in the user's preferences
    #         if news_row.topic not in preferences:
    #             continue  # Skip to the next neighbor

    #         # Display the news title and source
    #         st.sidebar.write(f"{news_row['title']}")
    #         st.sidebar.write(f"Source: {news_row['url']}")

    # embeddings = [np.array(i) for i in df.embedding]
    title_style = """
    <div style='font-size:15px; color:#white; margin-bottom:10px;'>
        {title}
    </div>
        """

    link_style = """
    <div style='margin-top:5px;'>
        <a href='{url}' style='color:#1f77b4; text-decoration:none; font-size:16px;'>
            Source
        </a>
    </div>"""


    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    k = 3
    # Perform a search to get the k nearest neighbors
    D, I = index.search(embeddings_np, k=k)
    other_topics_printed,Suggestions = True,True

    for news_id, counts in most_upvoted:
        # Check if the topic of the current news item is not in the user's preferences

        if counts['upvotes'] <= 0:
            continue
        
        for i in range(1, k):  # Start from 1 to skip the first neighbor (itself)
            # Fetch the news row corresponding to the current neighbor
            news_row = df.iloc[I[news_id][i]]
            print(I[news_id][i])
            

            

            if news_row.topic in preferences:
                if Suggestions:
                    st.sidebar.header('Suggestions:')
                    Suggestions = False
                
                st.sidebar.markdown(title_style.format(title=news_row['title']), unsafe_allow_html=True)
                st.sidebar.markdown(link_style.format(url=news_row['url']), unsafe_allow_html=True)
            else:
                if other_topics_printed:
                    st.sidebar.subheader("Similar articles from other topics:")
                    other_topics_printed = False

                st.sidebar.markdown(title_style.format(title=news_row['title']), unsafe_allow_html=True)
                st.sidebar.markdown(link_style.format(url=news_row['url']), unsafe_allow_html=True)
if __name__ == '__main__':
    main()
