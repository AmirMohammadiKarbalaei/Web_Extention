import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import defaultdict
import faiss
import pickle
from sitemaps_utils import Extract_todays_urls_from_sitemaps, process_news_data

# Function to fetch and process news data
@st.cache_data
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
    urls["sky"] = Extract_todays_urls_from_sitemaps(sky_news_sitemaps, namespaces, 'news:publication_date')

    bbc_topics_to_drop = {"pidgin", "hausa", "swahili", "naidheachdan"}
    df_BBC = process_news_data(urls, "bbc", bbc_topics_to_drop)

    # Uncomment and complete the following if Sky News processing is required
    # sky_topics_to_drop = {"arabic", "urdu"}
    # df_Sky = process_news_data(urls, "sky", sky_topics_to_drop)
    # return pd.concat([df_BBC, df_Sky])

    return df_BBC
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

    content_embedding = load_embeddings()
    embeddings = content_embedding[1]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    for news_id, counts in most_upvoted:
        if counts['upvotes'] > 0:
            D, I = index.search(embeddings, k=5)
            news_row = df_news.iloc[I[news_id][1]]
            st.sidebar.write(f"{news_row['Title']}")
            st.sidebar.write(f"Source: {news_row['Url']}")

if __name__ == '__main__':
    main()
