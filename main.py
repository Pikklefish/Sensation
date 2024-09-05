import pandas as pd
import re
import nltk
import sklearn
import time

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


## Import Dataset ##
training_file_path = 'C:/Users/work/trainingandtestdata/training.1600000.processed.noemoticon.csv'
test_file_path = 'C:/Users/work/trainingandtestdata/testdata.manual.2009.06.14.csv'

#Target: sentiemnt (0=negative, 2=netural, 4=positive)   Text: actual tweet
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df_train = pd.read_csv(training_file_path, encoding='latin-1', names=columns)
df_test = pd.read_csv(test_file_path, encoding='latin-1', names=columns)

## Process Data ##
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    #Remove URLS, mentions, tags etc ...
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)

    #lower case
    text = text.lower()

    #Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_words)

## Create a new cleaned text column ##
df_train['cleaned_text'] = df_train['text'].apply(preprocess_text)
df_test['cleaned_text'] = df_test['text'].apply(preprocess_text)

X_train = df_train['cleaned_text']
Y_train = df_train['target']

x_test = df_test['cleaned_text']
y_test = df_test['target']


## Vectorize the text ## 
vectorizer = TfidfVectorizer(max_features=5000)  #choose 5000 highest TF-IDF words and discard the rest (basically selected the top 5000 words that appear the most)
X_train_vect = vectorizer.fit_transform(X_train)
x_test_vect = vectorizer.fit_transform(x_test)

start_time = time.time()

## SVM model ##
svm_model = SVC(kernal = 'rbf', cache_size=2000, verbose = True)
print("training started")
svm_model = svm_model.fit(X_train_vect, Y_train)
print(f"training finished in {start_time}")
y_pred = svm_model.predict(x_test_vect)

print(classification_report(y_test, y_pred))