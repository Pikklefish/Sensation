from dependencies import *

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
