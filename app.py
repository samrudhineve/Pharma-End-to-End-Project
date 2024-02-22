# streamlit  web Application for Pharma drugs Predication - Samrudhi Neve
# Import necessary packages

import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Create a Header
st.set_page_config(page_title='Pharma Drugs - Samrudhi')

# Add a little to application in body
st.title("Pharma Drugs prediction - Samrudhi Neve")

# Take age as input from user
age = st.number_input("Age :", min_value=0, step=1)

# Take gender as input from user
gender = st.selectbox('Gender :', ('F','M'))

# Take BP as input from user
bp = st.selectbox('Blood Pressure :', ('HIGH', 'LOW', 'NORMAL'))

# Take Cholestrol as input from user
chol = st.selectbox('Cholesterol :', ('HIGH','NORMAL'))

# NA to K ratio
nak = st.number_input('Na to K Ratio :', min_value=0.000, step=0.001)

# Create a predict button
submit = st.button('Predict')

st.subheader('Predictions are : ')

# Create a function to predict the drug and probability of prediction
def predict_drugs(pipe_path, model_path):
    # Construct a dataframe from inputs
    dct = {'Age':[age],
           'Sex':[gender],
           'BP':[bp],
           'Cholesterol':[chol],
           'Na_to_K':[nak]}
    xnew = pd.DataFrame(dct)
    # Load the pipeline from the notebook folder
    with open(pipe_path, 'rb') as file1:
        pre = pickle.load(file1)
    #Load the model
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
    # Preprocessor the xnew
    xnew_pre = pre.transform(xnew)
    # Get predictions
    pred = model.predict(xnew_pre)
    # Get probability
    prob = model.predict_proba(xnew_pre)
    # Get max probability
    max_prob = np.max(prob)
    return pred, max_prob

# Logic if i press submit button
if submit:
    model_path =  "notebook/model.pkl"
    pipe_path = "notebook/pipe.pkl"
    pred, max_prob = predict_drugs(pipe_path, model_path)
    st.subheader(f'Predicted Drug is : {pred[0]}')
    st.subheader(f'Probability of Prediction : {max_prob:.4f}')
    st.progress(max_prob)