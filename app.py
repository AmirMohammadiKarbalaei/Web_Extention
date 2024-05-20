import streamlit as st
import pandas as pd
import concurrent.futures
from functools import lru_cache
from sitemaps_utils import *

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

    sky_topics_to_drop = {"arabic", "urdu"}
    df_Sky = process_news_data(urls, "sky", sky_topics_to_drop)

    return df_BBC#pd.concat([df_BBC, df_Sky])

st.cache_resource
def track_interaction(news_id, action):
    # Here, you could implement your tracking mechanism
    # For simplicity, let's just print the interaction
    print(f"User interacted with news article {news_id} - {action}")

def streamlit_print_topic_counts(data_frame: pd.DataFrame, section_name: str):
    """
    Function displays the count of each topic within a data frame,
    along with the total count. It's useful for summarizing topic
    distribution in a section.
    """
    st.subheader(section_name)
    topic_counts = data_frame['Topic'].value_counts()
    st.write(topic_counts)
    st.write(f"Total: {topic_counts.sum()}")

# Streamlit app
def main():
    st.title('News App')

    df_news = fetch_and_process_news_data()
    streamlit_print_topic_counts(df_news, 'Todays Topic Distribution')

    # Display topics
    topics = df_news['Topic'].unique()
    selected_topic = st.selectbox('Select a topic:', topics)

    # Filter news articles by selected topic
    selected_news = df_news[df_news['Topic'] == selected_topic]

    # Display news articles for selected topic
    st.subheader(f'Latest {selected_topic} News')
    for index, row in selected_news.iterrows():
        st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                <h4>{row['Title']}</h4>
                <p><a href="{row['Url']}" target="_blank">Read more</a></p>
                <div style="display: flex; justify-content: space-between;">
                    <button onclick="window.location.href='?upvote_{index}'">👍 Upvote</button>
                    <button onclick="window.location.href='?downvote_{index}'">👎 Downvote</button>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add interaction buttons with unique keys
    if st.button("👍 Upvote", key=f"upvote_{index}"):
        track_interaction(index, 'Upvoted')
    if st.button("👎 Downvote", key=f"downvote_{index}"):
        track_interaction(index, 'Downvoted')


if __name__ == '__main__':
    main()
