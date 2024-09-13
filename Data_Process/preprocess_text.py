from dependencies import *

def preprocess_text(text, lemmatizer, stop_words):
    try: 
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
    
    except Exception as e:
        logging.error(f"An unexpected error occurred in PREPROCESS_TEXT.py: {e}")
        st.error(f"Error occured in PREPROCESS_TEXT.py: {e}")
        return None

