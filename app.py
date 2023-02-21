import streamlit as st
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie
import math

# Test git
# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('customers.pkl', 'rb'))


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


st.title("Product Analytics")
st.subheader("Customer Segmentation")

lottie_coding = load_lottiefile("/Users/ssekar3/Desktop/QuickDocs/Projects/Customer_Segmentation/seg.json")
st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",
    height=None,
    width=None,
    key=None
)

# Age
age = st.number_input('Age of the Customer')

# Income
Annual_Income = st.number_input('Annual Income (k$) of the customer')

# Spending Score
Spending_score = st.number_input('Spending Score (1-100)')

# Gender
Gender = st.selectbox('Gender', ['Male', 'Female'])

if st.button('Segment Customer'):
    # query
    Male = None
    Female = None
    if Gender == 'Male':
        Male = 1
        Female = 0
    else:
        Female = 1
        Male = 0

    query = np.array([age, Annual_Income, Spending_score, Male, Female])

    query = query.reshape(1, 5)
    segment = pipe.predict(query)[0]
    if segment == 0:
        st.title("The customer belongs to Low Income - Low Spending segment")
    elif segment == 1:
        st.title("The customer belongs to High Income - High Spending segment")
    elif segment == 2:
        st.title("The customer belongs to Average Income - Average Spending segment")
    elif segment == 3:
        st.title("The customer belongs to High Income - Low Spending segment ")
    else:
        st.title("The customer belongs to Low Income - High spending segment")
