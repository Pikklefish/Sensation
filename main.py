import pandas as pd
import re
import nltk
import sklearn
import time
import joblib

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
from sklearn.model_selection import GridSearchCV

## Import Dataset ##
file_path = 'C:/Users/work/Sensation/reduced_tweet_dataset.csv'

#Target: sentiemnt (0=negative, 4=positive)   Text: actual tweet
df = pd.read_csv(file_path, encoding='latin-1')

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
df['cleaned_text'] = df['text'].apply(preprocess_text)

##Split data
target_0 = df[df['target'] == 0]
target_4 = df[df['target'] == 4]

train_0, test_0 = train_test_split(target_0, test_size=0.2, random_state=42)   # Split target 0 into training (80%) and testing (20%)
train_4, test_4 = train_test_split(target_4, test_size=0.2, random_state=42)


df_train = pd.concat([train_0, train_4])   # Combine the training data from both target groups
df_test = pd.concat([test_0, test_4])

X_train = df_train['cleaned_text']
Y_train = df_train['target']

x_test = df_test['cleaned_text']
y_test = df_test['target']

## Vectorize the text ## 
vectorizer = TfidfVectorizer(max_features=5000)  #choose 5000 highest TF-IDF words and discard the rest (basically selected the top 5000 words that appear the most)
print("Vectorization started")
start_time = time.time()
X_train_vect = vectorizer.fit_transform(X_train)
x_test_vect = vectorizer.transform(x_test)
print(f"Vectorization finished in {time.time() - start_time} seconds")


## Set Parameter ##
param_grid = {'C': [0.01, 0.1, 1, 10]}

## SVM model ##
svm_model = SVC(kernel = 'rbf', cache_size=1000, verbose = True)
grid_search = GridSearchCV(svm_model, param_grid, cv=5, verbose=2)
print("training started")
start_time = time.time()
grid_search.fit(X_train_vect, Y_train)
print(f"Training finished in {time.time() - start_time} seconds")
print(f"Best C value: {grid_search.best_params_['C']}")

best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(x_test_vect)

print(classification_report(y_test, y_pred))

# Save the model
model_file_path = "svm_model.pkl"
joblib.dump(best_svm_model, model_file_path)
print(f"Model saved to {model_file_path}")
