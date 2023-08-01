from selenium import webdriver
import pandas as pd
import numpy as np
import datetime
from lxml import etree
import langid
import re
from sklearn.metrics.pairwise import cosine_similarity
import requests
import textwrap
import json



from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('stopwords')
import string


def clean_text(text):
    # Remove punctuation marks from the text
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

# Assuming article_main_body is a list of articles




def Extract_todays_urls_from_BBC_sitemap(sitemaps:list,namespaces):
    """
    This function extracts the URLs and Last Modified
      and Title from XML sitemaps for today's date
    
    """
    sitemap_data = {}
    for sitemap in sitemaps:
        
        driver = webdriver.Chrome()

        # Load the XML sitemap
        driver.get(sitemap)
        sitemap_xml = driver.page_source

        # Parse the XML sitemap using lxml
        root = etree.fromstring(sitemap_xml.encode('utf-8'))

        # Get today's date
        today = datetime.date.today().isoformat()
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        urls = root.findall('.//sitemap:url', namespaces=namespace)

        # Extract lastmod, URL, and news title for each <url> element with today's last modified date
        for url in urls:
            lastmod = url.findtext('.//sitemap:lastmod', namespaces=namespace)
            loc = url.findtext('.//sitemap:loc', namespaces=namespace)
            news_title = url.findtext('.//news:title', namespaces=namespaces)
            
            if lastmod.startswith(today):
                sitemap_data[loc] = {
                    'Last Modified': lastmod,
                    'Title': news_title

                }

        # Close the Selenium driver
        driver.quit()
        return sitemap_data
    


def Extract_todays_urls_from_skysitemap(sitemaps: list, namespaces):
    sitemap_data = {}
    for sitemap in sitemaps:
        driver = webdriver.Chrome()

        # Load the XML sitemap
        driver.get(sitemap)
        sitemap_xml = driver.page_source

        # Parse the XML sitemap using lxml
        root = etree.fromstring(sitemap_xml.encode('utf-8'))

        # Get today's date
        today = datetime.date.today().isoformat()
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9', 'news': 'http://www.google.com/schemas/sitemap-news/0.9'}

        urls = root.findall('.//sitemap:url', namespaces=namespace)

        # Extract lastmod, URL, and news title for each <url> element with today's last modified date
        for url in urls:
            lastmod = url.findtext('.//news:publication_date', namespaces=namespace)
            loc = url.findtext('.//sitemap:loc', namespaces=namespace)
            news_title = url.findtext('.//news:title', namespaces=namespace)
            
            if lastmod.startswith(today):
                sitemap_data[loc] = {
                    'Last Modified': lastmod,
                    'Title': news_title
                }


        # Close the Selenium driver
        driver.quit()
    
        
    return sitemap_data





def is_english_sentence(sentence):
    """
    This function takes a sentence as input and uses the langid 
    library to classify the language of the sentence and returns True 
    if the sentence is in english.
    """
    lang, confidence = langid.classify(sentence)
    return lang == 'en'


def print_topic_counts(data_frame, section_name):
    print(f"------ {section_name} ------")
    topic_counts = data_frame['Topic'].value_counts()
    print(topic_counts.to_string())
    print(f"Total        {topic_counts.sum()}")
    print()



def Extract_sentences_from_urls(urls:list,driver=webdriver.Chrome()):
    articles = {}
    for index,url in enumerate(urls):
        
        main_content = []
    
       
        driver.get(url)

        page_source = driver.page_source
        tree = etree.HTML(page_source)

        for sentence in tree.findall(".//p"):
            main_content.append(sentence.text)

        articles[f"{url}"] = main_content

        driver.quit()
        print(f"{index} / {len(urls) - 1} ")
        print("Article has been Extracted")

    return articles



def preprocess_title(title):
    title = re.sub(r"[^a-zA-Z0-9\s]", "", title)  # Remove special characters
    title = title.lower()  # Convert to lowercase
    return title



def calculate_similarity(title1, title2,model = 'distilbert-base-nli-stsb-mean-tokens'):
    embedding_title1 = model.encode([title1])
    embedding_title2 = model.encode([title2])
    similarity_score = cosine_similarity(embedding_title1, embedding_title2)[0][0]
    return similarity_score



def request_sentences_from_url_(url):
    article_body = []
    
    # Fetch the web page's HTML content using requests
    response = requests.get(url)
    if response.status_code == 200:
        page_source = response.text
        tree = etree.HTML(page_source)
        if tree is not None:
            article_element = tree.find(".//article")
            if article_element is not None:
                # Get the outer HTML of the element using lxml's tostring()
                outer_html = etree.tostring(article_element, encoding='unicode')
                # Remove tags using the remove_tags() function
                article_body = remove_elements(outer_html)
        print("Article has been Extracted")
    else:
        print(f"Failed to fetch the web page. Status Code: {response.status_code}")
    
    return article_body



def remove_elements(input_string):
    pattern = r"<.*?>"
    
    # Use re.sub() to remove all matches of the pattern with an empty string
    cleaned_string = re.sub(pattern, "", input_string)
    return cleaned_string



def Text_wrap(Text:str,width:int):
    wrapped_string = textwrap.fill(Text, width=width)
    print(wrapped_string)
    return wrapped_string

def Save_to_json(file_name:str,data:any):    
    with open(f'{file_name}.json', 'w') as f:
            json.dump(data, f)
    print(f'File {file_name}.json has been saved')



def Open_json(file_name:str):    
    with open(file_name, 'r') as f:
            file = json.load(f)
    return file