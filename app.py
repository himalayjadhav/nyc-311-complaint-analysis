import streamlit as st

st.title("NYC Complaint Classifier")

st.write("Upload your complaint description below:")

text_input = st.text_input("Complaint Description")

if st.button("Predict"):
    st.write("You typed:", text_input)
    # This is where your ML prediction would go
