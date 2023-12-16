# Import necessary libraries
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pickle

# Load tokenizer and pre-trained model
with open("./model/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

model = tf.keras.models.load_model("./model/model.h5")

# Dictionary to map class indices to emotions
class_dict = {0: 'fear', 1: 'sadness', 2: 'anger', 3: 'joy'}

# Dictionary containing GIF URLs for each emotion
gif_dict = {
    'anger': 'https://media.giphy.com/media/l1J9u3TZfpmeDLkD6/giphy.gif',
    'fear': 'https://media.giphy.com/media/bEVKYB487Lqxy/giphy.gif',
    'joy': 'https://media.giphy.com/media/12XTNObsY1pWQU/giphy.gif',
    'sadness': 'https://media.giphy.com/media/9Y5BbDSkSTiY8/giphy.gif'
}

# Function to predict emotion based on input text
def predict_emotion(text):
    tokenized_input = tokenizer.texts_to_sequences([text])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(tokenized_input, truncating='post', padding='post', maxlen=50)
    output = model.predict(padded_input)
    predicted = class_dict[int(tf.argmax(output, axis=1).numpy()[0])]
    return predicted, output

# Main function for Streamlit app
def main():
    st.title("Emotion Classifier")
    st.subheader("Classify emotion in text")

    # Input form to receive user text
    with st.form(key="input_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")
        
    if submit_text:
        # Define columns for layout
        col1, col2 = st.columns(2)

        # Predict emotion and display results
        prediction, probability = predict_emotion(raw_text)

        with col1:
            confidence = f"{np.max(probability)*100:.2f}%"
            st.subheader(f"Prediction: {str.capitalize(prediction)} ({confidence})")
            st.image(gif_dict[prediction])
            
        with col2:
            st.subheader("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=class_dict.values())
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            # Create and display bar chart of prediction probabilities
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
    
# Run Streamlit app
if __name__ == "__main__":
    main()