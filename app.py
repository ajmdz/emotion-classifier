import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib

# pipe_lr = joblib.load(open("models/emotion_classifier_pipeline.pkl"))
with open("model/emotion_classifier_pipeline.pkl", "rb") as pipeline_file:
    pipe_lr = joblib.load(pipeline_file)

gif_dict = {
    'anger': 'https://media.giphy.com/media/l1J9u3TZfpmeDLkD6/giphy.gif',
    'disgust': 'https://media.giphy.com/media/xUA7bcUzBNSLC8Zy3C/giphy.gif',
    'fear': 'https://media.giphy.com/media/bEVKYB487Lqxy/giphy.gif',
    'joy': 'https://media.giphy.com/media/12XTNObsY1pWQU/giphy.gif',
    'neutral': 'https://media.giphy.com/media/XbgZvND0TzFMUFobmI/giphy.gif',
    'sadness': 'https://media.giphy.com/media/9Y5BbDSkSTiY8/giphy.gif',
    'shame': 'https://media.giphy.com/media/yjGdFXbm8KpXF5Xqco/giphy.gif',
    'surprise': 'https://media.giphy.com/media/oYtVHSxngR3lC/giphy.gif'

}

def predict_emotion(text):
    results = pipe_lr.predict([text])
    return results[0]

def get_prediction_proba(text):
    results = pipe_lr.predict_proba([text])
    return results

# emotions_emoji_dict

def main():
    st.title("Emotion Classifier")
    st.subheader("Classify emotion in text")
    

    with st.form(key="input_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")
        
    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotion(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            confidence = f"{np.max(probability)*100:.2f}%"
            st.subheader(f"Prediction: {str.capitalize(prediction)} ({confidence})")
            st.image(gif_dict[prediction])
            
        with col2:
            st.subheader("Prediction Probability")
            # st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            # st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
    

if __name__ == "__main__":
    main()

# streamlit run app.py