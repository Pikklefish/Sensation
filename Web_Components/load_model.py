from dependencies import *


def load_model( uploaded_model):
    if uploaded_model:
        try:
            # Load user-provided model from file
            model = joblib.load(uploaded_model)
            st.write("Model loaded successfully!")
            return model
        except Exception as e:
                st.write(f"Error loading the model: {e}")
    else:
        st.write("Please upload a pre-trained model.")
