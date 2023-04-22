import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_csv(path):
    df=pd.read_csv(path)
    return df

def load_labelencoder():
    lab=LabelEncoder()
    return lab


def load_model():
    with open('./model/randforest','rb') as f:
        model=pickle.load(f)
    return model
df=load_csv('./data/salaries.csv')

lab=load_labelencoder()
k=lab.fit_transform(df.company.unique())
j=lab.fit_transform(df.job.unique())
d=lab.fit_transform(df.degree.unique())

model=load_model()


if __name__=='__main__':
    st.write('U can check if one is earning more than 100k or not')

    company=st.selectbox('Select Company:',(df.company.unique()))
    if company=='google':
        company=k[0]
    if company=='abc pharma':
        company=k[1]
    if company=='facebook':
        company=k[2]
    job=st.selectbox('Select Position:',(df.job.unique()),key='uodcol')
    if job=='sales executive':
        job=k[0]
    if job=='business manager':
        job=k[1]
    if job=='computer programmer':
        job=k[2]
    degree=st.selectbox('Select Degree:',(df.degree.unique()),key='doias')
    if degree=='bachelors':
        degree=k[0]
    if degree=='masters':
        degree=k[1]
    button=st.button(label='Predict')
    if button:
        prediction=model.predict([[company,job,degree]])
        st.write('Your prdiction is :',prediction[0])
