from dependencies import *

def vectorizer_text(X_train, x_test, vectorizer):
    try:
        print("Vectorization started")
        start_time = time.time()
        X_train_vect = vectorizer.fit_transform(X_train)
        x_test_vect = vectorizer.transform(x_test)
        print(f"Vectorization finished in {time.time() - start_time} seconds")

        return X_train_vect, x_test_vect
    
    except Exception as e:
        st.error(f"Error occured in VECTORIZER_TEXT.py: {e}")