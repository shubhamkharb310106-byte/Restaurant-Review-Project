import streamlit as st
import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK stopwords
nltk.download('stopwords')

# Load the trained model
model = joblib.load("my_model.pkl")

# Load the saved CountVectorizer
cv = joblib.load("count_vectorizer.pkl")

# Create Porter Stemmer object
ps = PorterStemmer()

# Text preprocessing function
def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    return review

# -------------------------------
# Streamlit User Interface
# -------------------------------

st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️"
)

st.title("🍽️ Restaurant Review Sentiment Analysis")

st.write("Enter a restaurant review and click **Predict**.")

review = st.text_area(
    "Enter Review",
    placeholder="Example: The food was amazing and the staff was very friendly."
)

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_review = preprocess(review)

        review_vector = cv.transform([cleaned_review]).toarray()

        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")