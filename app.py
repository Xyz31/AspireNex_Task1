import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Movie Genre Predictor", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    train_file_path = "train_data.txt"
    return pd.read_csv(train_file_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')

@st.cache_resource
def train_model(data):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(data['Description'])
    y = data['Genre']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logreg_classifier = LogisticRegression(max_iter=1000)
    logreg_classifier.fit(X_train, y_train)
    
    y_pred = logreg_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return tfidf_vectorizer, logreg_classifier, accuracy

def predict_genre(title, description, vectorizer, model):
    input_data = [description]
    input_data_tfidf = vectorizer.transform(input_data)
    predicted_genre = model.predict(input_data_tfidf)
    return predicted_genre[0]

# Load data and train model
data = load_data()
vectorizer, model, accuracy = train_model(data)

# UI
st.markdown('<p class="big-font">Movie Genre Predictor</p>', unsafe_allow_html=True)

st.sidebar.markdown('<p class="medium-font">Model Information</p>', unsafe_allow_html=True)
st.sidebar.write(f"Model Accuracy: {accuracy:.2%}")
st.sidebar.write(f"Total Movies in Dataset: {len(data)}")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="medium-font">Enter Movie Details</p>', unsafe_allow_html=True)
    title = st.text_input("Movie Title")
    description = st.text_area("Movie Description")
    
    if st.button("Predict Genre"):
        if title and description:
            predicted_genre = predict_genre(title, description, vectorizer, model)
            st.success(f"The predicted genre for '{title}' is: {predicted_genre}")
        else:
            st.warning("Please enter both title and description.")

with col2:
    st.markdown('<p class="medium-font">Genre Distribution</p>', unsafe_allow_html=True)
    genre_counts = data['Genre'].value_counts()
    fig = px.pie(values=genre_counts.values, names=genre_counts.index, title="Genre Distribution")
    st.plotly_chart(fig)

st.markdown('<p class="medium-font">Sample Movies</p>', unsafe_allow_html=True)
sample_movies = data.sample(5)
st.table(sample_movies[['Title', 'Genre']])

st.markdown('<p class="medium-font">Word Cloud of Movie Descriptions</p>', unsafe_allow_html=True)
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(description for description in data.Description)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)