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
import textwrap
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)





#nltk.download('wordnet')
#nltk.download('stopwords')



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
    df_news = pd.DataFrame.from_dict(urls[source_name], orient='index').reset_index().rename(columns={'index': 'Url'})
    
    # Remove non-English entries
    df_news = df_news[df_news['Title'].apply(is_english_sentence)]
    
    # Extract topics from 'Url'
    df_news['Topic'] = df_news['Url'].str.extract(r'com/([^/]+)/')[0]
    # edge_1 = df_news['Url'].str.extract(r'/([^/]+)/([^/]+)/([^/]+)/([^/]+)/')
    # df_news.drop(index=edge_1[(edge_1[1] == "live") | (edge_1[2] == "live") | (edge_1[3] == "live")].index, inplace=True)

    
    # Drop rows with unwanted topics
    df_news = df_news[~df_news['Topic'].isin(topics_to_drop)].reset_index(drop=True)
    
    # Print topic counts
    print_topic_counts(df_news, source_name)
    
    return df_news

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



def is_english_sentence(sentence:str):
    """
    This function takes a sentence as input and uses the langid 
    library to classify the language of the sentence and returns True 
    if the sentence is in english.
    """
    lang, confidence = langid.classify(sentence)
    return lang == 'en'


def print_topic_counts(data_frame:pd.DataFrame, section_name:str):

    """
    function displays the count of each topic within a data frame, 
    along with the total count. It's useful for summarizing topic 
    distribution in a section.
    """
    print(f"------ {section_name} ------")
    topic_counts = data_frame['Topic'].value_counts()
    print(topic_counts.to_string())
    print(f"Total        {topic_counts.sum()}")
    print()


def preprocess_title(title:str):
    """
    This function removes special characters 
    and returns lowercased input string
    """
    title = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    return title.lower() 



def calculate_similarity(title1:str, title2:str,model:str = 'distilbert-base-nli-stsb-mean-tokens'):
    """
    This function measures the similarity 
    between two input titles using a pre-trained sentence 
    embedding model. It encodes the titles into numerical vectors, 
    calculates the cosine similarity between them, and returns a 
    single similarity score. 
    """
    embedding_title1 = model.encode([title1])
    embedding_title2 = model.encode([title2])
    similarity_score = cosine_similarity(embedding_title1, embedding_title2)[0][0]
    return similarity_score






def remove_elements(input_string: str):
    """
    This function removes all HTML tags and their content from the input string.
    """
    # Parse the input string as HTML
    soup = BeautifulSoup(input_string, 'html.parser')
    
    # Remove all tags and their content
    cleaned_string = soup.get_text(separator=' ', strip=True)
    
    return cleaned_string


def request_sentences_from_urls(urls, timeout=20):
    """
    Extracts the main article content from a list of URLs using `requests` and `lxml`.
    Returns a dictionary containing article bodies for each URL.
    Handles fetch errors gracefully and returns None for failed URLs.
    """

    articles_dict = {}

    for idx, url in enumerate(list(urls.Url), start=1):
        if (idx-1)%10 ==0:
            logging.info(f"\nProcessing URL {idx-1}/{len(urls)}")

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise exception for HTTP errors
        except requests.RequestException as e:
            logging.error(f"Failed to fetch the web page: {e}")
            continue
        except TimeoutError:
            logging.error(f"Timeout occurred while fetching URL: {url}")
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
                logging.warning("No article content found on the page.")
                continue
        except Exception as e:
            logging.error(f"Error extracting article content: {e} URL: {url}")
            continue

    return articles_dict



def Text_wrap(Text:str,width:int):
    """
    This function wraps text within a specified
      width for better printing visibility.
    
    """
    wrapped_string = textwrap.fill(Text, width=width)
    print(wrapped_string)
    return wrapped_string


def Save_to_json(file_name:str,data:any):
    """
     This Function Saves  
     given data into a json file .
    """   
    with open(f'{file_name}.json', 'w') as f:
            json.dump(data, f)
    print(f'File {file_name}.json has been saved')



def Open_json(file_name:str):
    """
    This Function Opens a Saved 
    json file and return the data.
    
    
    """    
    with open(file_name, 'r') as f:
            file = json.load(f)
    return file