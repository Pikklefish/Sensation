from dependencies import *

# ----Page Configuration----
st.set_page_config(page_title="Sensation", page_icon= "	:page_with_curl:", layout="wide")

# ----Header Section----
with st.container():
    st.subheader("This program will analyze the tone of your text :wave:")
    st.title("Sentiment Analysis ML model")

# Use local CSS

st.sidebar.title("Model Selection")
uploaded_model=st.sidebar.file_uploader("Upload a pre-trained model", type=['pkl'])


user_input = st.text_input("Enter text for analysis:")
if st.button("Analyze Text"):
    if user_input:
        st.write(f"Analyzing: {user_input}")

        processed_text = preprocess_text(user_input, lemmatizer, stop_words)

        user_input_vect = vectorizer_text_app(user_input, vectorizer_app)

        model = load_model(uploaded_model)
        prediction = model.predict(user_input_vect)
        if prediction == 0:
            st.write("🤖 The sentiment of the input text is **Negative**.")
        else:
            st.write("🤖 The sentiment of the input text is **Positive**.")
    else:
        st.write("Please enter some text to analyze.")
