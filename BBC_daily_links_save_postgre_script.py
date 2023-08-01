import psycopg2
from datetime import date
from sitemaps_utils import *




BBC_news_sitemaps = ["https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
                     "https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml",
                     "https://www.bbc.com/sitemaps/https-sitemap-com-news-3.xml"]


namespaces = {
    'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
    'news': 'http://www.google.com/schemas/sitemap-news/0.9'
}

BBC_today_urls = Extract_todays_urls_from_BBC_sitemap(BBC_news_sitemaps,namespaces)


data_frame_BBC = pd.DataFrame.from_dict(BBC_today_urls, orient='index')
data_frame_BBC["Url"]= data_frame_BBC.index
data_frame_BBC.reset_index(drop=True, inplace=True)


##Removing none english entries
for idx,title in  enumerate(data_frame_BBC.Title):
    if not is_english_sentence(title):
        data_frame_BBC.drop(idx,inplace=True)

data_frame_BBC['Topic'] = data_frame_BBC['Url'].str.split('com/').str[1].str.split('/').str[0]
topics_to_drop = ["pidgin", "hausa", "swahili", "naidheachdan"]

data_frame_BBC.drop(data_frame_BBC[data_frame_BBC['Topic'].isin(topics_to_drop)].index, axis=0, inplace=True)
data_frame_BBC.reset_index(drop=True,inplace=True)


all_content = []
for index,url in enumerate(data_frame_BBC.Url):
    main_body = request_sentences_from_url_(url)
    if main_body is not None:
        all_content.append(main_body)
    else:
        all_content.append([])
data_frame_BBC["Article content"] = all_content



##database credentials
db_credentials = {
    'dbname': 'Web_Extention',
    'user': 'postgres',
    'password': '2080',
    'host': 'localhost',
    'port': '5432'
}


connection = psycopg2.connect(**db_credentials)

# Creating a cursor to interact with the database
cursor = connection.cursor()
for _, row in data_frame_BBC.iterrows():
    insert_query = "INSERT INTO bbc_daily_links (last_modified, title, url, topic,article_content) VALUES (%s, %s, %s, %s, %s);"
    cursor.execute(insert_query, (row["Last Modified"], row["Title"], row["Url"], row["Topic"], row["Article content"]))


connection.commit()
cursor.close()
connection.close()
