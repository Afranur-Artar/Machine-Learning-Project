import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



st.title("Startup Success ML Project")
st.write("This App predicts startup success")
st.balloons()

df=pd.read_csv("startup data.csv")


df.drop(["Unnamed: 6"],axis=1, inplace=True)
df.drop(["Unnamed: 0"], axis=1, inplace=True)
df.drop(["id"], axis=1, inplace=True)
df.drop(["state_code.1"], axis=1, inplace=True)
df.drop(["closed_at"], axis=1, inplace=True)


df_new=df[["state_code","city","name","labels","founded_at","first_funding_at","last_funding_at","age_first_funding_year","age_last_funding_year","age_first_milestone_year","age_last_milestone_year","relationships","funding_rounds","funding_total_usd","milestones","category_code","has_angel","avg_participants","is_top500","status"]]
df_new=pd.get_dummies(df_new, columns=["status"], drop_first=True)


df_new.founded_at=pd.to_datetime(df_new.founded_at)
df_new.first_funding_at=pd.to_datetime(df_new.first_funding_at)
df_new.last_funding_at=pd.to_datetime(df_new.last_funding_at)


df_new["age_first_funding_year"]=np.abs(df_new["age_first_funding_year"])
df_new["age_last_funding_year"]=np.abs(df_new["age_last_funding_year"])
df_new["age_first_milestone_year"]=np.abs(df_new["age_first_milestone_year"])
df_new["age_last_milestone_year"]=np.abs(df_new["age_last_milestone_year"])

df_new["age_first_milestone_year"].fillna((df_new["age_first_milestone_year"].mean()), inplace=True)
df_new["age_last_milestone_year"].fillna((df_new["age_last_milestone_year"].mean()), inplace=True)

df_new["is_CA"]=df.is_CA
df_new["is_NY"]=df.is_NY
df_new["is_MA"]=df.is_MA
df_new["is_TX"]=df.is_TX
df_new["is_otherstate"]=df.is_otherstate
df_new["is_software"]=df.is_software
df_new["is_web"]=df.is_web
df_new["is_mobile"]=df.is_mobile
df_new["is_enterprise"]=df.is_enterprise
df_new["is_advertising"]=df.is_advertising
df_new["is_gamesvideo"]=df.is_gamesvideo
df_new["is_ecommerce"]=df.is_ecommerce
df_new["is_biotech"]=df.is_biotech
df_new["is_consulting"]=df.is_consulting
df_new["is_othercategory"]=df.is_othercategory
df_new["has_VC"]=df.has_VC
df_new["has_roundA"]=df.has_roundA
df_new["has_roundB"]=df.has_roundB
df_new["has_roundC"]=df.has_roundC
df_new["has_roundD"]=df.has_roundD

cols = [col for col in df_new if col != 'status_closed'] + ['status_closed'] 
df_new=df_new[cols]

st.sidebar.header("Exploratory Data Analysis")
option1 = st.sidebar.selectbox(
    'Which category do you want to see?',
     df_new['category_code'].unique())

for i in range(len(df_new["category_code"].unique())):
    if option1==df_new["category_code"][i]:
        st.write("Here is you choosed dataset")
        df_new[df_new["category_code"]==option1]
    
       
    
    
option2 = st.sidebar.selectbox(
    'Which plot do you want to visualize?',
     ("Barplot","Histogram","Scatter plot"))

# VISUALIZING
import plotly.figure_factory as ff
import altair as alt


if option2=="Barplot":
    if st.checkbox('Show Plot'):
        st.subheader('Number of Category Type')
        st.bar_chart(df_new["category_code"])
    
    
if option2=="Scatter plot":
    if st.checkbox('Show Plot'):
        c = alt.Chart(df_new).mark_circle().encode(x='age_first_funding_year', y='age_first_milestone_year')
        st.subheader("Relationship Between Funding & Milestone")
        st.altair_chart(c, use_container_width=True)
    
    
if option2=="Histogram":
    if st.checkbox('Show Plot'):
        x1=(df_new["age_first_funding_year"])
        x2=(df_new["age_last_funding_year"])
        x3=(df_new["age_first_milestone_year"])
        x4=(df_new["age_last_milestone_year"])
        hist_data=[x1,x2,x3,x4]
        group_labels=["First Funding","Last Funding","First Milestone","Last Milestone"]
        st.subheader("Frequency of Continuous Variables")
        fig = ff.create_distplot(hist_data, group_labels)
        st.plotly_chart(fig, use_container_width=True)

    
    
    
    
    

# MACHINE LEARNING ALGORITHM
st.sidebar.header("Machine Learning Algorithms")
option3 = st.sidebar.selectbox(
    'Which algorithm do you want to see?',
     ("Logistic Regression","Random Forest","XGBOOST"))    
    
st.subheader("User Input Parameters")
if option3=="Logistic Regression":
    def user_input_features():
        penalty=st.sidebar.selectbox("Penalty",("l1","l2"))
        solver=st.sidebar.selectbox("Solver", ("lbfgs","liblinear"))
        C=st.sidebar.selectbox("C", ([10**x for x in range(-3,3,1)]))
        data={"penalty":penalty,
              "solver":solver,
              "C":C
             }
        features=pd.DataFrame(data, index=[0])
        return features
    df=user_input_features()
    st.write(df)
    
    X = df_new.select_dtypes(exclude="O").drop(["status_closed","founded_at","first_funding_at","last_funding_at","age_first_funding_year",
                                              "age_last_funding_year","age_first_milestone_year","age_last_milestone_year"], axis=1)
    
    Y=df_new["status_closed"]
    
    X=StandardScaler().fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20, random_state=42)
    
    log_reg=LogisticRegression()
    log_reg.fit(X_train,Y_train)
    y_test_pred2=log_reg.predict(X_test)
    y_train_pred2=log_reg.predict(X_train)
    
    st.subheader("Accuracy of the Y_test:")
    st.write(accuracy_score(y_test_pred2,Y_test))
    st.subheader("Accuracy of the Y_train:")
    st.write(accuracy_score(y_train_pred2,Y_train))
    st.subheader("Confusion matrix:")
    st.write(confusion_matrix(Y_test, y_test_pred2))
    st.subheader("Classification report:") 
    st.write(classification_report(Y_test,y_test_pred2))
    

    
    
#RANDOM FOREST
if option3=="Random Forest":
    def user_input_features1():
        n_estimators=st.sidebar.slider("n_estimators", 10,40,10)
        criterion=st.sidebar.selectbox("criterion", ("gini","entropy"))
        max_depth=st.sidebar.slider("max_depth", 1,4,1)
        data={"n_estimators":n_estimators,
              "criterion":criterion,
              "max_depth":max_depth
             }
        features=pd.DataFrame(data, index=[0])
        return features
    df=user_input_features1()
    st.write(df)
    
    X1 = df_new.select_dtypes(exclude="O").drop(["status_closed","founded_at","first_funding_at","last_funding_at","age_first_funding_year",
                                              "age_last_funding_year","age_first_milestone_year","age_last_milestone_year"], axis=1)
    
    Y1=df_new["status_closed"]
    
    X1=StandardScaler().fit_transform(X1)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1,test_size=0.20, random_state=42)
    
    forest=RandomForestClassifier()
    forest.fit(X1_train, Y1_train)
    y1_test_forest_pred=forest.predict(X1_test)
    y1_train_forest_pred=forest.predict(X1_train)
                                       
    st.subheader("Accuracy of the random forest test")
    st.write(accuracy_score(y1_test_forest_pred,Y1_test))
    st.subheader("Accuracy of the random forest train")
    st.write(accuracy_score(y1_train_forest_pred,Y1_train))
    st.subheader("Confusion matrix")
    st.write(confusion_matrix(Y1_test,y1_test_forest_pred))
    st.write("Classification Report")
    st.header(classification_report(Y1_test, y1_test_forest_pred))
    
    
    
 
    
    
#XGBOOST
if option3=="XGBOOST":    
    X2 = df_new.select_dtypes(exclude="O").drop(["status_closed","founded_at","first_funding_at","last_funding_at","age_first_funding_year",
                                              "age_last_funding_year","age_first_milestone_year","age_last_milestone_year"], axis=1)
    
    Y2=df_new["status_closed"]
    
    X2=StandardScaler().fit_transform(X2)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2,test_size=0.20, random_state=42)
    
    xgb_model=XGBClassifier(objective='binary:logistic',
                       random_state=42,
                       eval_metric="auc")
    
    xgb_model.fit(X2_train,Y2_train, early_stopping_rounds=5, eval_set=[(X2_test,Y2_test)])
    y2_test_xgboost_pred=xgb_model.predict(X2_test)
    
    st.subheader("Accuracy of XGBOOST ")
    st.write(accuracy_score(Y2_test, y2_test_xgboost_pred))
                                       
                                     