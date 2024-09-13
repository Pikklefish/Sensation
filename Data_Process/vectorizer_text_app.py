from dependencies import *

def vectorizer_text_app(text, vectorizer_app):
    try:
        text_vect = vectorizer_app.transform([text])

        return text_vect
    
    except Exception as e:
        logging.error(f"An unexpected error occurred in VECTORIZER_TEXT_APP.py: {e}")
        st.error(f"Error occured in VECTORIZER_TEXT_APP.py: {e}")
        return None