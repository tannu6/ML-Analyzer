import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,r2_score,confusion_matrix

st.set_page_config(page_title="SmartML Analyzer",page_icon="✌️")
st.title("SmartML Analyzer 😎")

file=st.file_uploader("Upload CSV File",type=["csv"])
if file is None:
    st.stop()

df=pd.read_csv(file)
df=df.fillna(df.mean(numeric_only=True))
df=df.fillna("Unknown")

st.subheader("📄 Data")
st.write(df)

target=st.sidebar.selectbox("Select Target Column",df.columns)
test_size=st.sidebar.slider("Test Size",0.1,0.5,0.2)

df=df.dropna(subset=[target])

x=df.drop(columns=[target])
y=df[target]

x=pd.get_dummies(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size)

model_name=st.selectbox("Select Model",["Linear Regression","Multiple Linear Regression","Polynomial Regression","KNN","Decision Tree"])

is_classification=False

if model_name in ["Linear Regression","Multiple Linear Regression"]:
    model=LinearRegression()
    model.fit(x_train,y_train)
    predict=model.predict(x_test)

elif model_name=="Polynomial Regression":
    best_score=-1
    for d in range(1,5):
        poly=PolynomialFeatures(degree=d)
        x_train_poly=poly.fit_transform(x_train)
        x_test_poly=poly.transform(x_test)
        model=LinearRegression()
        model.fit(x_train_poly,y_train)
        pred=model.predict(x_test_poly)
        score=r2_score(y_test,pred)
        if score>best_score:
            best_score=score
            best_degree=d
            predict=pred
    st.write("Best Degree:",best_degree)

elif model_name=="KNN":
    is_classification=True
    best_acc=0
    for k in range(1,11):
        model=KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
        acc=accuracy_score(y_test,pred)
        if acc>best_acc:
            best_acc=acc
            best_k=k
            predict=pred
    st.write("Best K:",best_k)

elif model_name=="Decision Tree":
    is_classification=True
    best_acc=0
    for d in range(1,11):
        model=DecisionTreeClassifier(max_depth=d)
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
        acc=accuracy_score(y_test,pred)
        if acc>best_acc:
            best_acc=acc
            best_depth=d
            predict=pred
    st.write("Best Depth:",best_depth)

st.subheader("📊 Performance")

if is_classification:
    acc=accuracy_score(y_test,predict)
    rec=recall_score(y_test,predict,average="macro")
    f1=f1_score(y_test,predict,average="macro")
    cm=confusion_matrix(y_test,predict)
    st.write("Confusion Matrix")
    st.write(cm)
    st.write({"Model":model_name,"Accuracy":acc,"Recall":rec,"F1 Score":f1})
else:
    score=r2_score(y_test,predict)
    st.write({"Model":model_name,"R2 Score":score})

st.subheader("📊 Data Visualization")

col1=st.selectbox("Select X column",df.columns)
col2=st.selectbox("Select Y column",df.columns)

chart_type=st.selectbox("Select Chart Type",["Bar Chart","Line Chart","Scatter Plot","Pie Chart"])

if chart_type=="Bar Chart":
    plt.bar(df[col1],df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("Bar Chart")
    st.pyplot(plt)

elif chart_type=="Line Chart":
    plt.plot(df[col1],df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("Line Chart")
    st.pyplot(plt)

elif chart_type=="Scatter Plot":
    sns.scatterplot(x=df[col1],y=df[col2])
    plt.title("Scatter Plot")
    st.pyplot(plt)

elif chart_type=="Pie Chart":
    sizes=df[col1].value_counts()
    labels=sizes.index
    plt.pie(sizes,labels=labels,autopct="%1.1f%%")
    plt.title("Pie Chart")
    st.pyplot(plt)

st.subheader("🔥 Correlation Heatmap")

corr=df.corr(numeric_only=True)
sns.heatmap(corr,annot=True)
st.pyplot(plt)