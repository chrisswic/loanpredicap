import pandas as pd
import streamlit as st
# import seaborn as sns 
# import matplotlib.pyplot as plt
import numpy as np 
import sklearn
import warnings 
warnings.filterwarnings('ignore')
import joblib


data = pd.read_csv('loan_testData.csv')

# load the model
model = joblib.load(open('loan_data.pkl', 'rb'))


#--------------- streamlit development starts -------------------
st.markdown("<h1 style = ' color: #053B50'> SWIZ LOAN COMPANY.CO</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'top-margin: 0rem; color: #BB2525'>BUILT By CHRISTOPHER SWIZ DATA SCIENCTIST</h1>", unsafe_allow_html=True)

st.title('LOAN PREDICTION')
st.write('built by CHRISTOPHER SWIZ')


st.write('PLEASE ENTER YOUR USERNAME')
USERNAME = st.text_input(' PLEASE ENTER YOUR USERNAME:')
PASSWORD = st. text_input('PLEASE ENTER YOUR PASSWORD:')
if st.button('RESET PASSWORD'):
    st.success(f"WELCOME {USERNAME}.please enjoy usage")

elif st.button('SUBMIT PASSWORD'): 
   st.success(f"welcome {PASSWORD}.swiz")

st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<h2 style = 'top-margin: 0rem;text-align: center; color: #A2C579'>LOAN PREDICTION</h1>", unsafe_allow_html=True)

st.write("In the domain of financial lending, accurately predicting whether a loan applicant is likely to default on their loan is crucial for minimizing risk and ensuring sound financial decision-making. The challenge lies in effectively assessing various factors such as credit history, income, loan amount, and loan term to determine the likelihood of repayment. The absence of a reliable prediction model leads to increased default rates, financial losses, and a negative impact on both lenders and borrowers. Therefore, developing an efficient loan prediction model is imperative for mitigating risks associated with loan defaults and fostering a sustainable lending ecosystem.")


# heat_map = plt.figure(figsize = (14,7)) #---------------------------------------------------create a heat map plot
correlation_data = data[['Dependents',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Credit_History', 'Loan_Status']] #----sele data fr corelation

# sns.heatmap(correlation_data.corr(), annot = True, cmap = 'BuPu')
# st.write(heat_map)
data.drop('Unnamed: 0', axis = 1, inplace = True)
st.write(data.sample(10))

st.sidebar.image('abc5f0442efe2d9f4755dfa2adb35034.png', width= 300 , caption = f"WELCOME {USERNAME}",use_column_width= True)

st.markdown("<br>", unsafe_allow_html= True)

st.sidebar.write('please decide a loan amount')
input_style = st.sidebar.selectbox('pick your preferred number for prediction', ['Slider input','Number input'])
if input_style == 'Slider input':
    Dependent = st.sidebar.slider('Dependents', data['Dependents'].min(), data['Dependents'].max())
    Applicant = st.sidebar.slider('ApplicantIncome', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())  
    Coapplicant = st.sidebar.slider('CoapplicantIncome', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
    Loan = st.sidebar.slider('LoanAmount', data['LoanAmount'].min(), data['LoanAmount'].max())
    Loan_Amount = st.sidebar.slider('Loan_Amount_Term', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())
    Credit_History = data['Credit_History'].unique()
    Credit_History  =  st.sidebar.selectbox('Select Your State Model', Credit_History)

else:
    Dependent = st.sidebar.number_input('Dependents', data['Dependents'].min(), data['Dependents'].max())
    Applicant = st.sidebar.number_input('ApplicantIncome', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())  
    Coapplicant = st.sidebar.number_input('CoapplicantIncome', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
    Loan = st.sidebar.number_input('LoanAmount', data['LoanAmount'].min(), data['LoanAmount'].max())
    Loan_Amount = st.sidebar.number_input('Loan_Amount_Term', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())
    Credit_History = data['Credit_History'].unique()
    Credit_History  =  st.sidebar.selectbox('Select Your State Model', Credit_History)


st.subheader("Your Inputted Data")
input_var = pd.DataFrame([{  'Dependents':Dependent,
       'ApplicantIncome': Applicant, 'CoapplicantIncome':Coapplicant, 'LoanAmount':Loan,
       'Loan_Amount_Term':Loan_Amount,}])
st.write(input_var)

st.markdown("<br>", unsafe_allow_html= True)
tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])

with tab1:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_var)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')


with tab2:
    st.subheader('Model Interpretation')
    st.write(f"LoanAmount = {model.intercept_.round(2)} + {model.coef_[0].round(2)}  + {model.coef_[1].round(2)} ApplicantIncome + {model.coef_[2].round(2)} Credit_History")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected LoanAmount for a startup is {model.intercept_}")

    st.markdown(f"- For every additional loan collected , this amount is expected to be paid as the loan company interest ${model.coef_[0].round(2)}  ")

    st.markdown(f"- For every delay this total amount would be deducted from your account ${model.coef_[1].round(2)}  ")

    st.markdown(f"- for delays an extra sum would be deducted ${model.coef_[2].round(2)}  ")