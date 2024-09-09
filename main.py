from dependencies import *

# ## Import Dataset ##
# file_path = 'C:/Users/work/Sensation/reduced_tweet_dataset.csv'

# #Target: sentiemnt (0=negative, 4=positive)   Text: actual tweet
# df = pd.read_csv(file_path, encoding='latin-1')

# ## Create a new cleaned text column ##
# df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))

# ## Process Text
# X_train, Y_train, x_test, y_test = split_data(df)
# X_train_vect, x_test_vect = vectorizer_text(X_train, x_test, vectorizer)
# joblib.dump(vectorizer,'vectorizer.pkl')


# ## SVM model ##
# svm_model = SVC(kernel = 'linear', cache_size=2000, verbose = True)
# print("training started")
# start_time = time.time()
# svm_model = svm_model.fit(X_train_vect, Y_train)
# print(f"Training finished in {time.time() - start_time} seconds")
# y_pred = svm_model.predict(x_test_vect)

# print(classification_report(y_test, y_pred))

# # Save the model
# model_file_path = "svm_model.pkl"
# joblib.dump(svm_model, model_file_path)
# print(f"Model saved to {model_file_path}")










def load_or_train_model( uploaded_model):
    if uploaded_model:
        # Load user-provided model from file
        model = joblib.load(uploaded_model)
        st.write("Model loaded successfully!")
        return model


##Streamlit UI components
st.title("Sentiment Analysis ML model")

user_input = st.text_input("Enter text for analysis:")

st.sidebar.title("Model Selection")
uploaded_model=st.sidebar.file_uploader("Upload a pre-trained model", type=['pkl'])

if st.button("Analyze Text"):
    if user_input:
        st.write(f"Analyzing: {user_input}")

        processed_text = preprocess_text(user_input, lemmatizer, stop_words)

        user_input_vect = vectorizer_text_app(user_input, vectorizer_app)

        model = load_or_train_model(uploaded_model)
        prediction = model.predict(user_input_vect)
        if prediction == 0:
            st.write("🤖 The sentiment of the input text is **Negative**.")
        else:
            st.write("🤖 The sentiment of the input text is **Positive**.")
    else:
        st.write("Please enter some text to analyze.")
