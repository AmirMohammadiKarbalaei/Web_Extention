import mysql.connector  # Ensure you use mysql.connector for MySQL connections
from datetime import date
import pandas as pd
from sitemaps_utils import Extract_todays_urls_from_sitemaps, is_english_sentence, request_sentences_from_urls_async

# Define BBC news sitemaps
BBC_news_sitemaps = [
    "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
    "https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml",
    "https://www.bbc.com/sitemaps/https-sitemap-com-news-3.xml"
]

# Define XML namespaces
namespaces = {
    'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
    'news': 'http://www.google.com/schemas/sitemap-news/0.9'
}
today = date.today()

# Extract today's urls from BBC sitemaps
BBC_today_urls = Extract_todays_urls_from_sitemaps(BBC_news_sitemaps, namespaces,today)

# Create a DataFrame from the extracted urls
data_frame_BBC = pd.DataFrame.from_dict(BBC_today_urls, orient='index')
data_frame_BBC["url"] = data_frame_BBC.index
data_frame_BBC.reset_index(drop=True, inplace=True)

# Remove non-English entries
for idx, title in enumerate(data_frame_BBC.title):
    if not is_english_sentence(title):
        data_frame_BBC.drop(idx, inplace=True)

# Extract topics and filter out non-relevant topics
data_frame_BBC['topic'] = data_frame_BBC['url'].str.split('com/').str[1].str.split('/').str[0]
topics_to_drop = ["pidgin", "hausa", "swahili", "naidheachdan"]
data_frame_BBC.drop(data_frame_BBC[data_frame_BBC['topic'].isin(topics_to_drop)].index, axis=0, inplace=True)
data_frame_BBC.reset_index(drop=True, inplace=True)

# Fetch article content for each url
all_content = []
for index, url in enumerate(data_frame_BBC.url):
    main_body = request_sentences_from_urls_async(url)
    all_content.append(main_body if main_body is not None else [])
data_frame_BBC["Article content"] = all_content

# Database credentials
db_credentials = {
    'database': 'Web_Extention',
    'user': 'root',  # Adjust username if different
    'password': '2080',
    'host': 'localhost',
    'port': '3306'  # Default MySQL port
}

# Connect to the database
connection = mysql.connector.connect(**db_credentials)

# Create a cursor to interact with the database
cursor = connection.cursor()
for _, row in data_frame_BBC.iterrows():
    insert_query = """
    INSERT INTO bbc_daily_links (last_modified, title, url, topic, article_content)
    VALUES (%s, %s, %s, %s, %s);
    """
    cursor.execute(insert_query, (row["Last Modified"], row["title"], row["url"], row["topic"], row["Article content"]))

# Commit the transaction and close the connection
connection.commit()
cursor.close()
connection.close()
