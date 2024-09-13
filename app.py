from dependencies import *

def main():
    
    uploaded_model, user_input = ui_setup()

    if st.button("Analyze Text"):
        if user_input:
            st.write(f"Analyzing: {user_input}")

            processed_text = preprocess_text(user_input, lemmatizer, stop_words)

            user_input_vect = vectorizer_text_app(processed_text, vectorizer_app)

            model = load_model(uploaded_model)
            prediction = model.predict(user_input_vect)

            if prediction == 0:
                st.error("🤖 Negative Sentiment Detected!")
            else:
                st.success("🤖 Positive Sentiment Detected!")
                st.balloons()  # Trigger balloons for positive sentiment

        else:
            st.write("Please enter some text to analyze.")

if __name__=="__main__":
    main()
