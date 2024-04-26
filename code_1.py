import streamlit as st
import re
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
loaded_model = pickle.load(open("trained_model_nlp.sav", 'rb'))

# Load the TfidfVectorizer
loaded_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

def preprocess_tweet(tweet):
    # Preprocess the tweet
    new_tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    new_tweet = new_tweet.lower()
    new_tweet = new_tweet.split()
    new_tweet = [port_stem.stem(word) for word in new_tweet if not word in stopwords.words("english")]
    new_tweet = ' '.join(new_tweet)
    return new_tweet

def predict_sentiment(tweet):
    # Preprocess the tweet
    processed_tweet = preprocess_tweet(tweet)
    # Vectorize the tweet
    vectorized_tweet = loaded_vectorizer.transform([processed_tweet])
    # Predict the sentiment
    prediction = loaded_model.predict(vectorized_tweet)
    return prediction[0]

# Streamlit app
def main():
    st.title("Sentiment Analysis")
    st.write("Enter your tweet below:")

    user_input = st.text_input("Input tweet")

    if st.button("Predict"):
        if user_input:
            prediction = predict_sentiment(user_input)
            if prediction == 1:
                st.write("Positive sentiment")
            else:
                st.write("Negative sentiment")
        else:
            st.write("Please enter a tweet to predict.")

if __name__ == "__main__":
    main()
