# -*- coding: utf-8 -*-

# %% 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # If you're using it 
from dotenv import load_dotenv
import os
from openai import OpenAI

                                                   
# %% 
 #LOADING DATASET
df = pd.read_csv('311_Service_Requests_from_2010_to_Present_20250718.csv')
df.head()
df.info()
print(df.shape[1])  # Gives you just the number of columns


# %% 

df["Created Date"] = pd.to_datetime(df["Created Date"], errors='coerce')
df["Closed Date"] = pd.to_datetime(df["Closed Date"], errors='coerce')
df.head()

# %% 
df = df.dropna(subset=["Complaint Type", "Created Date", "Borough"])
df.head()

# %%
df[["Complaint Type", "Created Date", "Borough"]].isna().sum()

#--------------------------------------------------------------------------------------------------

# %% 

print("Before:", df.shape[1])
print("Columns before drop:", df.columns.tolist())

# Drop columns
df = df.drop(columns=[
  "Incident Address", "Location", "Cross Street 1",
  "Cross Street 2", "Intersection Street 1", "Intersection Street 2",
  "Street Name", "Due Date", "Resolution Action Updated Date"
], errors='ignore')

print("After:", df.shape[1])
print("Columns after drop:", df.columns.tolist())

#--------------------------------------------------------------------------------------------------

# %% 

df["Response Time (hrs)"] = (df["Closed Date"] - df["Created Date"]).dt.total_seconds() / 3600
df["Hour"] = df["Created Date"].dt.hour
df["Weekday"] = df["Created Date"].dt.day_name()
df["Month"] = df["Created Date"].dt.month
df.head()
df.info() 
df.describe()

#--------------------------------------------------------------------------------------------------


# %% 

df[["Complaint Type", "Created Date", "Borough"]].head(10)

#--------------------------------------------------------------------------------------------------

# %% 
df.sort_values(by="Created Date").head(10)

#--------------------------------------------------------------------------------------------------

# %% 
from IPython.display import display

# Summary of the DataFrame
print("🔢 Shape:", df.shape)


#--------------------------------------------------------------------------------------------------

# %% 

# Missing values in key columns
print("\n❗ Missing Values:")
print(df[["Complaint Type", "Created Date", "Borough"]].isna().sum())

#--------------------------------------------------------------------------------------------------

# %% 
# Sample of key columns (first 10 rows)
print("\n👀 Sample Data (First 10 rows):")
display(df[["Complaint Type", "Created Date", "Borough"]].head(10))

#--------------------------------------------------------------------------------------------------

# %% 

# View sorted dates to confirm time filtering
print("\n📅 Date Range:")
print("Min:", df["Created Date"].min())
print("Max:", df["Created Date"].max())

#--------------------------------------------------------------------------------------------------

# %% 
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nData Types:\n", df.dtypes)

#--------------------------------------------------------------------------------------------------

# %% 
df['Complaint Type'].value_counts().head(10).plot(kind='bar', figsize=(15, 3), title='Top 10 Complaint Types')
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------
# %% 
df['Date'] = df['Created Date'].dt.date
df['Date'].value_counts().sort_index().plot(figsize=(15, 3), title='Daily Complaint Volume')
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------
# %% 

df['Hour'] = df['Created Date'].dt.hour
df['Hour'].value_counts().sort_index().plot(kind='bar', figsize=(15, 3), title='Complaints by Hour of Day')
plt.tight_layout()
plt.show()


#--------------------------------------------------------------------------------------------------
# %% 


pivot = df.pivot_table(index='Borough', columns='Complaint Type', aggfunc='size', fill_value=0)
plt.figure(figsize=(15,3))
sns.heatmap(pivot, cmap='Blues')

#--------------------------------------------------------------------------------------------------
# %% 

df['Borough'].value_counts().head(10).plot(
    kind='bar', figsize=(15, 3),
    title='Complaints by borough',
    color='skyblue'
    )

#--------------------------------------------------------------------------------------------------

# %% 
df['Date'] = df['Created Date'].dt.date  # Extract just the date (no time)
daily_counts = df['Date'].value_counts().sort_index()

daily_counts.plot(
    figsize=(15, 5),
    title='Daily Complaint Volume',
    xlabel='Year',
    ylabel='Number of Complaints',
    color='purple'
)

#--------------------------------------------------------------------------------------------------
# %% 
avg_resolution = df.groupby("Complaint Type")["Response Time (hrs)"].mean().sort_values(ascending=False)

avg_resolution.head(10).plot(
    kind='barh',
    figsize=(10, 6),
    title='Top 10 Complaint Types by Avg. Resolution Time',
    xlabel='Avg. Resolution Time (hrs)',
    color='teal'
)

#--------------------------------------------------------------------------------------------------
# %% 


# Create a pivot table
pivot = df.pivot_table(index='Complaint Type', columns='Borough', aggfunc='size', fill_value=0)

# Limit to top complaint types for clarity
top_types = df['Complaint Type'].value_counts().head(10).index
pivot = pivot.loc[top_types]

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Top Complaint Types by Borough')
plt.xlabel('Borough')
plt.ylabel('Complaint Type')
plt.show()

#--------------------------------------------------------------------------------------------------
# %% 

df['Day of Week'] = df['Created Date'].dt.day_name()

df['Day of Week'].value_counts().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
]).plot(
    kind='bar',
    figsize=(15, 3),
    color='coral',
    title='Complaints by Day of the Week'
)
# %% 
df.to_csv("311_cleaned.csv", index=False)


#--------------------------------------------------------------------------------------------------
# %% 

load_dotenv()  # 🔄 Load environment variables from .env

api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"              
)

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


#--------------------------------------------------------------------------------------------------

# %% 

test_complaint = "Noise complaint from construction during night hours."
category = classify_text_with_mistral(test_complaint)
print("Predicted Category:", category) 


#--------------------------------------------------------------------------------------------------

# %% 
top_complaints = df['Complaint Type'].value_counts().head(5).index
df_ml = df[df['Complaint Type'].isin(top_complaints)].copy()
df_ml["Complaint Type"].value_counts()


#--------------------------------------------------------------------------------------------------

# %% 


import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
custom_stopwords = set(stopwords.words('english')) - {'no', 'not', 'never'}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in custom_stopwords)
    return text

df_ml['cleaned_description'] = df_ml['Descriptor'].fillna("").apply(clean_text)
df_ml[['Descriptor', 'cleaned_description']].head()


#--------------------------------------------------------------------------------------------------

# %% 

"""**TEXT VECTORIZATION USING TF-IDF**
"""

from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF tool from scikit-learn

vectorizer = TfidfVectorizer()  # Initialize the vectorizer

# Transform cleaned descriptor text into a matrix of TF-IDF features
X = vectorizer.fit_transform(df_ml['cleaned_description'])  # X holds the numeric representation of the text

# Target variable: the complaint type we are trying to predict  
y = df_ml['Complaint Type']  # y holds the labels/classification targets

print(X.shape)  # Output: (rows = number of samples, columns = number of unique words/features)

#--------------------------------------------------------------------------------------------------

# %% 

"""**MODEL TRAINING & EVALUATION**"""

# Import necessary tools for training and evaluating the model
from sklearn.linear_model import LogisticRegression  # Our ML model
from sklearn.model_selection import train_test_split  # To split data into train/test sets
from sklearn.metrics import classification_report, accuracy_score  # To evaluate model performance

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model: print accuracy and detailed performance report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#--------------------------------------------------------------------------------------------------

# %% 

"""**Make Predictions with Our Model**"""

# Try predicting a new descriptor
new_description = ["no heating in the apartment"]
new_cleaned = [clean_text(desc) for desc in new_description]  # Preprocess just like training data
new_vectorized = vectorizer.transform(new_cleaned)  # Convert to numeric using same TF-IDF

# Make prediction
predicted = model.predict(new_vectorized)

print("Predicted Complaint Type:", predicted[0])

#--------------------------------------------------------------------------------------------------
# %% 
import pickle

# Save the trained logistic regression model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the fitted TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

    # Example inference
text = ["The vehicle is parked in my doorway"]
text_vector = vectorizer.transform(text)
prediction = model.predict(text_vector)
print(prediction)

