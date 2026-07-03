import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Student Performance Prediction System",
    page_icon="🎓",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = Path(__file__).parent / "model" / "student_model.pkl"

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        return model

    except FileNotFoundError:
        st.error("❌ Model file not found!")
        st.stop()

    except Exception as e:
        st.error(f"❌ Error loading model : {e}")
        st.stop()


model = load_model()

# ----------------------------
# Title
# ----------------------------

st.title("🎓 AI-Powered Student Performance Prediction")
st.write(
    """
This system predicts whether a student is **Low Risk**, **Medium Risk**,
or **High Risk** before the final examination and provides recommendations
for teachers.
"""
)

# ----------------------------
# Student Input
# ----------------------------

st.header("📋 Student Details")

col1, col2 = st.columns(2)

with col1:

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        80
    )

    quiz = st.slider(
        "Quiz Score",
        0,
        100,
        70
    )

    assignment = st.slider(
        "Assignment Score",
        0,
        100,
        75
    )

    internal = st.slider(
        "Internal Exam",
        0,
        100,
        72
    )

    previous = st.slider(
        "Previous Percentage",
        0,
        100,
        74
    )

with col2:

    participation = st.selectbox(
        "Participation",
        [1,2,3],
        format_func=lambda x:
        {
            1:"Low",
            2:"Medium",
            3:"High"
        }[x]
    )

    homework = st.slider(
        "Homework Completion (%)",
        0,
        100,
        80
    )

    behaviour = st.slider(
        "Behaviour Score",
        1,
        5,
        4
    )

    study = st.slider(
        "Study Hours / Day",
        0.0,
        8.0,
        2.5
    )

    parent = st.selectbox(
        "Parents Attended Meeting?",
        [0,1],
        format_func=lambda x:
        {
            0:"No",
            1:"Yes"
        }[x]
    )

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Risk"):

    input_data = pd.DataFrame({

        "Attendance":[attendance],

        "QuizScore":[quiz],

        "AssignmentScore":[assignment],

        "InternalExam":[internal],

        "PreviousPercentage":[previous],

        "Participation":[participation],

        "HomeworkCompletion":[homework],

        "BehaviourScore":[behaviour],

        "StudyHours":[study],

        "ParentMeeting":[parent]

    })

    prediction = model.predict(input_data)[0]

    st.divider()

    st.header("Prediction Result")

    if prediction == "High":

        st.error("🔴 HIGH RISK STUDENT")

    elif prediction == "Medium":

        st.warning("🟡 MEDIUM RISK STUDENT")

    else:

        st.success("🟢 LOW RISK STUDENT")

    st.subheader(f"Predicted Risk Level : {prediction}")

    # --------------------------------
    # Early Warning
    # --------------------------------

    st.header("🚨 Early Warning System")

    if prediction == "High":

        st.error(
            "Immediate teacher intervention is recommended."
        )

    elif prediction == "Medium":

        st.warning(
            "Monitor the student's progress every week."
        )

    else:

        st.success(
            "Student is performing well."
        )

    # --------------------------------
    # Recommendation System
    # --------------------------------

    st.header("📌 Personalized Recommendations")

    recommendations = []

    if attendance < 75:

        recommendations.append(
            "Improve classroom attendance."
        )

    if quiz < 50:

        recommendations.append(
            "Attend remedial quiz sessions."
        )

    if assignment < 50:

        recommendations.append(
            "Complete pending assignments."
        )

    if internal < 50:

        recommendations.append(
            "Extra preparation for internal examinations."
        )

    if study < 2:

        recommendations.append(
            "Increase study time to at least 2-3 hours daily."
        )

    if participation == 1:

        recommendations.append(
            "Participate more actively during lectures."
        )

    if homework < 70:

        recommendations.append(
            "Submit homework regularly."
        )

    if parent == 0:

        recommendations.append(
            "Arrange a parent-teacher meeting."
        )

    if len(recommendations) == 0:

        st.success(
            "Excellent performance. Keep maintaining the same consistency."
        )

    else:

        for rec in recommendations:

            st.write("✅", rec)