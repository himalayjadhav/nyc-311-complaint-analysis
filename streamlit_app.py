import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
import os
from openai import OpenAI

# --- Load Environment Variables --- #
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- Initialize OpenAI Client for LLM --- #
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# --- Load Cleaned Data --- #
df = pd.read_csv("311_cleaned.csv")

# --- Load Model and Vectorizer --- #
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# --- Preprocessing Function --- #
nltk.download('stopwords')
custom_stopwords = set(stopwords.words('english')) - {'no', 'not', 'never'}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(word for word in text.split() if word not in custom_stopwords)

# --- LLM Classification --- #
def classify_text_with_mistral(prompt_text):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies 311 complaints into categories."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# --- Streamlit UI --- #
st.title("📊 NYC 311 Complaints Classifier")
st.markdown("""This app allows you to classify 311 complaints using either a trained ML model or an AI model via LLM.
""")

# Select Method
method = st.radio("Choose Method:", ["ML Model (Logistic Regression)", "AI Model (LLM - Mistral)"])

# User Input
user_input = st.text_area("Enter Complaint Description:", "Illegal parking in front of my garage")

if st.button("Classify Complaint"):
    if method == "ML Model (Logistic Regression)":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"ML Model Prediction: {prediction}")

    elif method == "AI Model (LLM - Mistral)":
        ai_prediction = classify_text_with_mistral(user_input)
        st.success(f"LLM Prediction: {ai_prediction}")

# Optional: Show top complaint types from dataset
if st.checkbox("Show top complaint types in dataset"):
    top_complaints = df['Complaint Type'].value_counts().head(10)
    st.bar_chart(top_complaints)







