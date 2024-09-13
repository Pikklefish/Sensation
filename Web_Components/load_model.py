from dependencies import *


def load_model( uploaded_model):
    if uploaded_model:
        try:
            # Load user-provided model from file
            model = joblib.load(uploaded_model)
            st.write("Model loaded successfully!")
            return model
        except Exception as e:
                logging.error(f"An unexpected error occurred in UPLOADED_MODEL.PY: {e}")
                st.error(f"Error loading the model in UPLOADED_MODEL.PY: {e}")
                return None
    else:
        st.write("Please upload a pre-trained model.")
