import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Load the pre-trained model and tokenizer
model_path = os.path.join('models', 'model.keras')
model = tf.keras.models.load_model(model_path)

tokenizer_path = os.path.join('models', 'tokenizer.pickle')
tokenizer = pickle.load(open(tokenizer_path, 'rb'))

# Preprocess user input text
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=model.input_shape[1], padding='post', truncating='post')
    return padded

# Define a reset function
def reset_inputs():
    st.session_state.user_input = ""

# Initialize session state for the input field
if "user_input" not in st.session_state:
    reset_inputs()

# Title and description
st.title("IMDB Reviews Sentiment Classifier")
st.write("""
This application predicts whether a movie review is positive or negative. 
Type a review below and click 'Predict' to see the result.
""")

# Input field with session state
user_input = st.text_area("Write your review here:", height=200, key='user_input')

# Buttons for predict and clear
col1, col2 = st.columns(2)
with col1:
    predict_button = st.button('Predict')
with col2:
    clear_button = st.button('Clear', on_click=reset_inputs)

# Prediction logic
if predict_button:
    if user_input.strip():
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        confidence = prediction[0][0] if sentiment == "Positive" else 1 - prediction[0][0]

        # Display result
        st.subheader("Prediction Result:")
        if sentiment == "Positive":
            st.success(f"The review is **Positive**.")
        else:
            st.error(f"The review is **Negative**.")
    else:
        st.warning("Please write a review before clicking Predict!")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("""
This sentiment classifier is trained on the IMDB dataset and uses a deep learning model built with TensorFlow.
The application is designed to provide sentiment analysis for movie reviews.
""")

st.sidebar.header("How It Works")
st.sidebar.write("""
1. Enter a movie review in the input box.
2. Click the 'Predict' button to analyze the sentiment.
3. Use the 'Clear' button to reset the input field.
""")

st.sidebar.header("Developer Notes")
st.sidebar.write("""
- Built with Python, TensorFlow, Streamlit and pickle.
- Contact for questions or suggestions:
  - **Email**: ahmed.hamdii.kamal@gmail.com
  - **GitHub**: https://github.com/hamdii10
""")
