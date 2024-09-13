from dependencies import *

def ui_setup():
    try:
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

        return uploaded_model, user_input
    
    except Exception as e:
        logging.error(f"An unexpected error occurred in UI_SETUP.PY: {e}")
        st.error(f"Error loading UI in UI_SETUP.PY: {e}")
        return None

