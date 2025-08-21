import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import pandas as pd
import nltk
import matplotlib.pyplot as plt

# Set page configuration at the very top
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Download stopwords from NLTK if not already downloaded
# This is a robust way to ensure the data is present on the server
try:
    nltk.data.find('corpora/stopwords')
except (nltk.downloader.DownloadError, AttributeError):
    nltk.download('stopwords')

# This function handles text preprocessing


def text_preprocessing(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

# Use Streamlit's cache to load models only once


@st.cache_resource
def load_models():
    try:
        tfidf_vectorizer_title = joblib.load('tfidf_vectorizer_title.joblib')
        tfidf_vectorizer_text = joblib.load('tfidf_vectorizer_text.joblib')
        best_model = joblib.load('best_model.joblib')
        return tfidf_vectorizer_title, tfidf_vectorizer_text, best_model
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'tfidf_vectorizer_title.joblib', 'tfidf_vectorizer_text.joblib', and 'best_model.joblib' are in the same directory.")
        st.stop()


tfidf_vectorizer_title, tfidf_vectorizer_text, best_model = load_models()

# --------------------------------------------------------------------------------
# Sidebar for About Section
with st.sidebar:
    st.header("About This Project")
    st.write("This is a final master's project in Data Science focusing on the automatic detection of fake news using NLP techniques.")
    st.markdown("---")
    st.subheader("Methodology")
    st.write("The application uses a trained **Support Vector Machine (SVM)** model. The model classifies news articles based on their title and body text, which are transformed into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.")
    st.markdown("---")
    st.subheader("Model Performance")

    # Bar chart for model metrics
    metrics = {'Accuracy': 0.981, 'F1-Score': 0.97,
               'Precision': 0.98, 'Recall': 0.982}
    fig, ax = plt.subplots()
    bars = ax.bar(metrics.keys(), metrics.values(), color=[
        'skyblue', 'salmon', 'lightgreen', 'gold'])
    ax.set_ylim(0.9, 1.0)
    ax.set_title('Model Evaluation Metrics')

    # labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)

    st.markdown("---")
    st.write("Developed by: Telu Raju")

# --------------------------------------------------------------------------------
# Main Content
st.title("Fake News Detector 📰")
st.markdown("### A Master's Project in Data Science")

# Section 1: The Problem Statement and Context
st.header("The Threat of Misinformation")
st.markdown("""
Fake news can be extremely dangerous, as misinformation can impact one's daily life, from personal and professional decisions to financial status and family relationships. This danger stems from the way misinformation spreads and is consumed. As the **"stimulus and response"** framework shows, individuals can be passive recipients of false information, while the **"active audience"** approach reveals a more complex reality: people actively engage with and share news, sometimes regardless of its truth.
""")

col1, col2 = st.columns(2)
with col1:
    st.image("Interesting.JPG", caption="Stimulus and response framework & Active audience approach.",
             use_container_width=True)
with col2:
    st.image("digitalMediaAndMisInformation.PNG",
             caption="The lifecycle of digital media and misinformation.", use_container_width=True)
    st.markdown("""
    This behavior contributes to a rapid **lifecycle of digital misinformation**, where a false story, once posted, spreads widely long before its veracity can be challenged or confirmed. Before taking decisions, it's critical to verify information with valid sources. We urge you to ensure it's 100% true and correct, as your discernment is the key to combating this dangerous cycle.
    """)

# Section 2: Data Visualization
st.header("Dataset Overview")
st.write("This chart shows the distribution of fake and real news articles in the dataset used to train the model.")

# Create a bar chart for dataset distribution
dataset_counts = pd.DataFrame({
    'News Type': ['Fake', 'Real'],
    'Count': [23481, 21417]
})
st.bar_chart(dataset_counts.set_index('News Type'), color="#ffaa00")

# --------------------------------------------------------------------------------
# Section 3: The Interactive Demo
st.header("Live Demo: Classify a News Article")
st.write("Enter a news headline and body text below for prediction.")

# Input fields for the user
title_input = st.text_input("News Headline:")
body_input = st.text_area("News Body Text:", height=200)

if st.button("Predict"):
    if title_input and body_input:
        # Preprocess and vectorize the input
        processed_title = text_preprocessing(title_input)
        processed_body = text_preprocessing(body_input)

        title_vectorized = tfidf_vectorizer_title.transform([processed_title])
        body_vectorized = tfidf_vectorizer_text.transform([processed_body])

        # Combine the features into a single dataframe
        combined_features = pd.concat([pd.DataFrame(title_vectorized.toarray()),
                                       pd.DataFrame(body_vectorized.toarray())], axis=1)

        # Make the prediction and get confidence score
        prediction = best_model.predict(combined_features)[0]
        # LinearSVC does not provide a probability, so we use decision_function as a proxy for confidence.
        confidence_score = best_model.decision_function(combined_features)[0]

        # Display the result
        st.markdown("---")
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(
                f"✅ **REAL** news: The model is **{abs(confidence_score):.2f}** confident this is real.")
        else:
            st.error(
                f"❌ **FAKE** news: The model is **{abs(confidence_score):.2f}** confident this is fake.")
            st.warning(
                "🚨 **WARNING:** Please be cautious with this information. We recommend verifying this article with a trusted news source.")
    else:
        st.warning(
            "Please provide both a headline and body text to get a prediction.")

