# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:35:18 2021

@author: Acer
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff 
from PIL import Image
import os
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns 
st.set_page_config(page_title="Data_exploring",
                   page_icon=None, layout="wide")

st.title("CS Week7 Python FN", anchor=None)

st.text("web application with STreamlit")

st.sidebar.image(Image.open('fnlogo.jpg'))

df=pd.read_csv('loan_prediction.csv')
 
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df["Loan_Status"]=lb.fit_transform(df["Loan_Status"])

df.dropna()
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df[["Gender","Married","Self_Employed","Loan_Amount_Term"]]=imp.fit_transform(df[["Gender","Married","Self_Employed","Loan_Amount_Term"]])                 

menu=st.sidebar.selectbox("Let's start",
                          ["Mainpage","Data exploration", "modelling"])
if menu=="Mainpage":
    st.dataframe(df)
    st.header("Mainpage")
    st.image(Image.open('homepagephoto.png'), use_column_width="always")
    st.text("Company wants to automate the loan eligibility process provided while filling online application form.") 
    
    
    
    chart1 =px.pie(data_frame=df,
              title="Pie Chart",
              names="Gender",
              values="Loan_Status")
    st.plotly_chart(chart1, use_container_width=True)
    
    chart2 =px.bar(data_frame=df,    
                   x="Loan_Status",
                   y="LoanAmount",
                   color="Gender",
                   title="Scatter")
    st.plotly_chart(chart2, use_container_width=True)
    numeric_columns=df.select_dtypes(exclude=["object"])
    result=sns.pairplot(df,vars=list(numeric_columns))
    st.pyplot(result)
    
   
    
    
elif menu=="Data exploration":
    
        
    st.subheader("Description")
    def describeStat(df):
        df.describe().T
    res = describeStat(df)
    st.write("brief info", res)
    
    
    
    st.subheader("Null variables")
    null_df=df.isnull().sum()
    null_df.columns=["Columns", "Counts"]
    st.write(" ", null_df)
    
    st.subheader('Imbalance checking')
    st.bar_chart(df.iloc[:,-1].value_counts())
   
else:
   
    st.header("Let's see prediction")
    df=pd.get_dummies(df,drop_first=True)
    df_=df.drop(df.select_dtypes(include=['object']).columns,axis=1)
    df=pd.concat([df_,df],axis=1)
    
    st.subheader("Train_test splitting")
    from sklearn.model_selection import train_test_split
    y=df.Loan_Status
    X=df.drop("Loan_Status",axis=1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.23,random_state=42)
    st.markdown('X_train size = {0}'.format(X_train.shape))
    st.markdown('X_test size = {0}'.format(X_test.shape))
    st.markdown('y_train size = {0}'.format(y_train.shape))
    st.markdown('y_test size = {0}'.format(y_test.shape))
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
    
    st.subheader("Scaling")
    from sklearn.preprocessing import RobustScaler
    rc=RobustScaler().fit(X_train)
    X_train_scaled=rc.transform(X_train)
    X_test_scaled=rc.transform(X_test)
    st.markdown('Scaling')
    st.write(rc)
    
    
    st.subheader("Choose the model?")   
    model=st.selectbox(" ", ["Xgboost", "Catboost"])
    st.title('Congratulations Your Model is working')
    
    
    if model=="Xgboost":
        import xgboost as xgb
        model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
        model.fit(X_train_scaled, y_train,eval_metric=['auc'],verbose=True)
        
        cm=confusion_matrix(y_test,model.predict(X_test_scaled))
        st.markdown('Confusion Matrix')
        st.write(cm)

        acc = accuracy_score(y_test, model.predict(X_test_scaled)) 
        st.markdown("Accuracy Score = "+acc)
        st.write("Accuracy: ", acc)
                
    else:
        from catboost import CatBoostClassifier 
        model=CatBoostClassifier().fit(X_train, y_train)
        
        cm=confusion_matrix(y_test,model.predict(X_test_scaled))
        st.markdown('Confusion Matrix')
        st.write(cm)

        acc = accuracy_score(y_test, model.predict(X_test_scaled)) 
        st.markdown("Accuracy Score = "+acc)
        st.write("Accuracy: ", acc)    

    st.title("Thanks For using")