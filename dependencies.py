import pandas as pd
import nltk
import time
import joblib
import re
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


from Data_Process.preprocess_text import preprocess_text
from Data_Process.split_data import split_data
from Data_Process.vectorizer_text import vectorizer_text
from Data_Process.vectorizer_text_app import vectorizer_text_app

from Web_Components.load_model import load_model

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(max_features=5000)  #choose 5000 highest TF-IDF words and discard the rest (basically selected the top 5000 words that appear the most)

vectorizer_app = joblib.load('vectorizer.pkl')