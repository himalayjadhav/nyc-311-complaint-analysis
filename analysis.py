# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
    df['Date'] = df['Created Date'].dt.date
    df['Hour'] = df['Created Date'].dt.hour
    return df

def complaints_by_date(df):
    df['Date'].value_counts().sort_index().plot(figsize=(12, 3), title='Daily Complaint Volume')
    plt.tight_layout()
    plt.show()

def heatmap_by_borough(df):
    pivot = df.pivot_table(index='Borough', columns='Complaint Type', aggfunc='size', fill_value=0)
    plt.figure(figsize=(20,6))
    sns.heatmap(pivot, cmap='Blues')
    plt.tight_layout()
    plt.show()
