#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import pickle

import streamlit
import streamlit as st
print('done')
# loading in the model to predict on the data
pickle_in = open('majormodel.pkl', 'rb')
classifier = pickle.load(pickle_in)
print('done1')

def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(member_id, emp_length, loan_amnt, funded_amnt, funded_amnt_inv, sub_grade, int_rate, annual_inc, dti, mths_since_last_delinq, mths_since_last_record, open_acc, revol_bal, revol_util, total_acc, total_rec_int, mths_since_last_major_derog, last_week_pay, tot_cur_bal, total_rev_hi_lim, tot_coll_amt, term, recoveries):
    prediction = classifier.predict(
        [[member_id, emp_length, loan_amnt, funded_amnt, funded_amnt_inv, sub_grade, int_rate, annual_inc, dti, mths_since_last_delinq, mths_since_last_record, open_acc, revol_bal, revol_util, total_acc, total_rec_int, mths_since_last_major_derog, last_week_pay, tot_cur_bal, total_rev_hi_lim, tot_coll_amt, term, recoveries]])
    print(prediction)
    return prediction
print('done2')

# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("LOAN-PREDICTION")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Loan Prediction Classifier ML App (Major project) </h1> 
    </div> 
    """
#print('done3')
    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)
#print('done4')
    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    member_id= st.text_input("member_id", "Type Here")
    emp_length= st.text_input("emp_length", "Type Here")
    loan_amnt = st.text_input("loan_amnt", "Type Here")
    funded_amnt = st.text_input("funded_amnt", "Type Here")
    funded_amnt_inv = st.text_input("funded_amnt_inv", "Type Here")
    sub_grade = st.text_input("sub_grade", "Type Here")
    int_rate = st.text_input("int_rate ", "Type Here")
    Aannual_inc = st.text_input("annual_inc", "Type Here")
    dti = st.text_input("dti", "Type Here")
    mths_since_last_delinq = st.text_input("mths_since_last_delinq", "Type Here")
    mths_since_last_record = st.text_input("mths_since_last_record", "Type Here")
#     open_acc = st.text_input("open_acc", "Type Here")
#     revol_bal = st.text_input("revol_bal", "Type Here")
#     revol_util = st.text_input("revol_util", "Type Here")
#     total_acc = st.text_input("total_acc", "Type Here")
#     total_rec_int = st.text_input("total_rec_int", "Type Here")
#     mths_since_last_major_derog = st.text_input("mths_since_last_major_derog", "Type Here")
#     last_week_pay = st.text_input("last_week_pay", "Type Here")
#     tot_cur_bal = st.text_input("tot_cur_bal", "Type Here")
#     total_rev_hi_lim = st.text_input("total_rev_hi_lim", "Type Here")
#     tot_coll_amt = st.text_input("tot_coll_amt", "Type Here")
#     term = st.text_input("term", "Type Here")
#     recoveries = st.text_input("recoveries", "Type Here")
    result = ""
#print('done5')
    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(member_id, emp_length, loan_amnt, funded_amnt, funded_amnt_inv, sub_grade, int_rate, annual_inc, dti, mths_since_last_delinq, mths_since_last_record, open_acc, revol_bal, revol_util, total_acc, total_rec_int, mths_since_last_major_derog, last_week_pay, tot_cur_bal, total_rev_hi_lim, tot_coll_amt, term, recoveries)
    st.success('The output is {}'.format(result))

print('done6')

if __name__ == '__main__':
    print('done6')
    main()
    


# In[ ]:




