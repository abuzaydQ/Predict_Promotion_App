import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

st.title("Promotion Class Predictor")

st.write("This is to predict if the students will be able to pass their promotion exam")

studentExam = pd.read_csv("PromotedClass_dataset.csv")

features = ["Hours_Studied","Attendance","Assignments_Completed"]

X = studentExam[features]
y = studentExam["Result"]


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

st.sidebar.header("Enter your new records")
Studied_Hours = st.sidebar.slider("How many hours used for studying?", min_value=0.0, max_value=10.0, step=1.0, value=0.0)
Attendance_Score = st.sidebar.slider("What is your attendance scored?", min_value=0.0, max_value=100.0, step=5.0, value=0.0)
Assignment_Score = st.sidebar.slider("What is your score graded for your assignment?", min_value=0.0, max_value=10.0, step=1.0, value=0.0)


Current_StudentDf = pd.DataFrame({
    "Hours_Studied" : [Studied_Hours],
    "Attendance" : [Attendance_Score],
    "Assignments_Completed" : [Assignment_Score],
    
})

st.subheader("Student Info")
st.dataframe(Current_StudentDf)

if st.button("Run KNN classifier"):
    prediction=model.predict(Current_StudentDf)[0]

    if prediction == "Pass":
        st.success("The student will likely pass")

    else:
        st.error("The student will likely fail")