# 🗽 NYC 311 Complaint Analysis + AI/ML Prediction

This project analyzes NYC 311 service request data to identify public complaint trends and uses both Machine Learning (ML) and AI (LLM) to predict complaint types from text. Built using Python, Streamlit, and Groq’s Mistral model, this end-to-end project offers data insights and intelligent classification.

---

## 🚀 Project Overview

🧩 **Goal**: Help city officials and citizens understand major complaint patterns across NYC and intelligently classify complaints using AI/ML.

🎯 **Business Problem Solved**:
- Understand most common complaints by borough and time.
- Identify which types of complaints take the longest to resolve.
- Predict complaint types from raw descriptions using ML & AI.
- Provide an intelligent interface for faster triaging of issues.

---

## 🛠️ Tools & Technologies Used

| Tool        | Purpose |
|-------------|---------|
| **Pandas**  | Data processing |
| **Seaborn / Matplotlib** | EDA Visualizations |
| **Scikit-learn** | ML model training (Logistic Regression) |
| **OpenAI + Groq API** | LLaMA 3 (Mistral) for AI classification |
| **dotenv**  | Secure API key management |
| **Streamlit** | Frontend prediction app |
| **VS Code** | Development environment |

---

## 📂 Folder Structure

.
├── main.py # All analysis, ML, AI steps
├── streamlit_app.py # App to run AI/ML from browser
├── 311_cleaned.csv # Cleaned data used in the project
├── model.pkl # Trained Logistic Regression model
├── vectorizer.pkl # TF-IDF transformer
├── .env # API key (not pushed to GitHub)
├── /screenshots/ # Visuals for demo
└── README.md # This file

yaml
Copy
Edit

---

## 📈 Project Steps

### ✅ Step 1: Load & Clean Dataset
- Loaded NYC 311 CSV from recent data
- Converted dates, removed irrelevant columns
- Dropped missing values and created:
  - `Response Time (hrs)`
  - `Hour`, `Month`, `Weekday` features

### ✅ Step 2: EDA & Visualizations
- Top complaint types overall
- Daily complaint trends
- Heatmap: complaints by time/borough
- Bar charts for day, hour, borough, resolution time

### ✅ Step 3: AI Text Classification
- Used **Groq + Mistral (LLaMA 3)** via API
- Prompted LLM with complaint description to return category

### ✅ Step 4: ML Model
- Filtered top 5 complaint types
- Cleaned complaint descriptions with NLTK
- Vectorized using **TF-IDF**
- Trained **Logistic Regression** model
- Achieved ~80% accuracy on test set
- Saved model with `pickle`

### ✅ Step 5: Streamlit App
- Simple browser UI to:
  - Classify complaint text using ML
  - Get AI predictions using Groq API
  - Enter and test any user input

---

## 🧠 AI + ML in Action

| Technique | Role |
|----------|------|
| **ML (Logistic Regression)** | Learns from top 5 complaint types and classifies new text |
| **AI (LLM via Groq)** | Responds to natural language queries, classifies text intelligently |
| **TF-IDF** | Transforms text into numerical format for ML |
| **NLTK** | Preprocessing: stopwords removal, text cleaning |

---

## 📊 Sample EDA Screenshots

### 🔹 Streamlit App
!<img width="681" height="603" alt="nyc_ai" src="https://github.com/user-attachments/assets/d61e1bbd-d97d-4c85-a8dd-fa2729a9f98c" />
<img width="702" height="570" alt="nyc_ml" src="https://github.com/user-attachments/assets/363bb264-ff48-450d-b716-211b589262fe" />


### 🔹 Daily Complaint Trend
<img width="1490" height="290" alt="NYC_daily_complaint_volume" src="https://github.com/user-attachments/assets/946abdca-cc17-4620-9e73-5672f11a8c80" />


### 🔹 Top 10 Complaints
<img width="1218" height="553" alt="output2" src="https://github.com/user-attachments/assets/95e953ce-3ca5-42aa-b735-8a1f8a286fd1" />


---

## 💼 Real-World Value

Based on this project, NYC officials can:
1. Prioritize high-volume complaint types.
2. Identify slow-resolution issues and improve SLAs.
3. Spot peak hours/days for service demand.
4. Quickly triage new complaints using AI/ML prediction.
5. Make borough-specific decisions with confidence.

---

## 🔒 Environment Setup

Create a `.env` file with your API key:

GROQ_API_KEY= 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



## 📦 Run This Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Launch the Streamlit App
bash
streamlit run streamlit_app.py
🪄 Future Enhancements
Integrate map-based visualization (folium/plotly)

Add feedback loop to improve AI predictions

Deploy to Streamlit Cloud / Hugging Face Spaces

Allow CSV upload for batch classification



