import pandas as pd
import numpy as np
import pickle

import streamlit
import streamlit as st

# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(loan_id, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    prediction = classifier.predict(
        [[loan_id, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])
    print(prediction)
    return prediction


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("LOAN PREDICTION")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Loan Prediction Classifier ML App (Major project) </h1> 
    </div> 
    """

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    loan_id = st.text_input("loan_id", "Type Here")
    Gender = st.text_input("Gender", "Type Here")
    Married = st.text_input("Married ", "Type Here")
    Dependents = st.text_input("Dependents", "Type Here")
    Education = st.text_input("Education", "Type Here")
    Self_Employed = st.text_input("Self_Employed ", "Type Here")
    ApplicantIncome = st.text_input("ApplicantIncome", "Type Here")
    CoapplicantIncome = st.text_input("CoapplicantIncome", "Type Here")
    LoanAmount = st.text_input("LoanAmount", "Type Here")
    Loan_Amount_Term = st.text_input("Loan_Amount_Term", "Type Here")
    Credit_History = st.text_input("Credit_History", "Type Here")
    Property_Area = st.text_input("Property_Area", "Type Here")
    result = ""

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(loan_id, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area)
    st.success('The output is {}'.format(result))



if __name__ == '__main__':
    main()



