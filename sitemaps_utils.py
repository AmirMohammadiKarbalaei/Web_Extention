from selenium import webdriver
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





def Extract_todays_urls_from_BBC_sitemap(sitemaps:list,namespaces):
    """
    This function extracts the URLs and Last Modified
      and Title from XML sitemaps for today's date from BBC sitemap
    
    """
    sitemap_data = {}
    for sitemap in sitemaps:
        
        driver = webdriver.Chrome()

        driver.get(sitemap)
        sitemap_xml = driver.page_source

        root = etree.fromstring(sitemap_xml.encode('utf-8'))

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

        driver.quit()
        return sitemap_data
    


def Extract_todays_urls_from_skysitemap(sitemaps: list, namespaces):

    """
    This function extracts the URLs and Last Modified
      and Title from XML sitemaps for today's date from sky sitemap
    
    """
     

    sitemap_data = {}
    for sitemap in sitemaps:
        driver = webdriver.Chrome()

        driver.get(sitemap)
        sitemap_xml = driver.page_source

        root = etree.fromstring(sitemap_xml.encode('utf-8'))


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


        driver.quit()
    
        
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



def request_sentences_from_url_(url:str):
    """
    This  function extracts the main article content from a URL using 
    `requests` and `lxml`. It returns the article body as a list of sentences after 
    removing unnecessary tags. Handles fetch errors with an error message.
    """



    article_body = []
    response = requests.get(url)

    if  response.status_code != 200:
        print(f"Failed to fetch the web page. Status Code: {response.status_code}")
        return

    page_source = response.text
    tree = etree.HTML(page_source)
    if tree is not None:
        article_element = tree.find(".//article")
        if article_element is not None:
            outer_html = etree.tostring(article_element, encoding='unicode')
            article_body = remove_elements(outer_html)

    print("Article has been Extracted")

        
    
    return article_body



def remove_elements(input_string:str):
    """
    This function removes all data between < >.
    
    """

    pattern = r"<.*?>"
    cleaned_string = re.sub(pattern, "", input_string)
    return cleaned_string



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