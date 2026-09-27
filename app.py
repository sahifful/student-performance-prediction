import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "student_performance_model.joblib"
)

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "..",
    "database"
)

DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "predictions.db"
)


# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def initialize_database():

    os.makedirs(DATABASE_DIR, exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            prediction TEXT NOT NULL,
            pass_probability REAL NOT NULL,
            fail_probability REAL NOT NULL,
            prediction_time TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    username,
    prediction,
    pass_probability,
    fail_probability
):

    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            username,
            prediction,
            pass_probability,
            fail_probability,
            prediction_time
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        username,
        prediction,
        pass_probability,
        fail_probability,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_prediction_history():

    conn = sqlite3.connect(DATABASE_PATH)

    query = """
        SELECT
            id,
            username,
            prediction,
            pass_probability,
            fail_probability,
            prediction_time
        FROM predictions
        ORDER BY id DESC
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


# Initialize database
initialize_database()


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

try:

    import joblib

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("Unable to load the machine learning model.")
    st.code(str(e))
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""


# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.title("🎓 Student Performance Prediction System")

    st.subheader("Login")

    st.write(
        "Please enter your username and password."
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        login_button = st.button(
            "Login",
            use_container_width=True
        )

        if login_button:

            if username == "admin" and password == "admin123":

                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = "Admin"

                st.rerun()

            elif username == "user" and password == "user123":

                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = "User"

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    st.info(
        "Demo accounts: admin/admin123 or user/user123"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 Student Prediction")

st.sidebar.write(
    f"Logged in as: **{st.session_state.username}**"
)

st.sidebar.write(
    f"Role: **{st.session_state.role}**"
)

st.sidebar.divider()


if st.session_state.role == "Admin":

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Model Information",
            "Prediction History",
            "Prediction System"
        ]
    )

else:

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Prediction System"
        ]
    )


st.sidebar.divider()


if st.sidebar.button(
    "Logout",
    use_container_width=True
):

    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""

    st.rerun()


# =========================================================
# ADMIN DASHBOARD
# =========================================================

if menu == "Dashboard":

    st.title("📊 Admin Dashboard")

    st.write(
        "Overview of the Student Performance Prediction System."
    )

    # -----------------------------------------------------
    # LOAD PREDICTION HISTORY
    # -----------------------------------------------------

    history = get_prediction_history()

    total_predictions = len(history)

    if not history.empty:

        pass_predictions = (
            history["prediction"] == "Pass"
        ).sum()

        fail_predictions = (
            history["prediction"] == "Fail"
        ).sum()

        average_pass_probability = (
            history["pass_probability"].mean()
        )

        average_fail_probability = (
            history["fail_probability"].mean()
        )

    else:

        pass_predictions = 0
        fail_predictions = 0
        average_pass_probability = 0
        average_fail_probability = 0


    # -----------------------------------------------------
    # SYSTEM KPIs
    # -----------------------------------------------------

    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Dataset Records",
            "395"
        )

    with col2:

        st.metric(
            "Input Features",
            "30"
        )

    with col3:

        st.metric(
            "Models Tested",
            "3"
        )

    with col4:

        st.metric(
            "Final Model",
            "Random Forest"
        )


    # -----------------------------------------------------
    # PREDICTION KPIs
    # -----------------------------------------------------

    st.subheader("Prediction Activity")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Predictions",
            total_predictions
        )

    with col2:

        st.metric(
            "Predicted Pass",
            pass_predictions
        )

    with col3:

        st.metric(
            "Predicted Fail",
            fail_predictions
        )


    # -----------------------------------------------------
    # PREDICTION DISTRIBUTION
    # -----------------------------------------------------

    if not history.empty:

        st.divider()

        st.subheader(
            "Prediction Distribution"
        )

        distribution_df = pd.DataFrame({
            "Prediction": [
                "Pass",
                "Fail"
            ],
            "Number of Predictions": [
                pass_predictions,
                fail_predictions
            ]
        })

        col1, col2 = st.columns(2)

        with col1:

            st.bar_chart(
                distribution_df.set_index(
                    "Prediction"
                )
            )

        with col2:

            st.dataframe(
                distribution_df,
                use_container_width=True,
                hide_index=True
            )


        # -------------------------------------------------
        # PROBABILITY ANALYSIS
        # -------------------------------------------------

        st.subheader(
            "Average Prediction Probability"
        )

        probability_df = pd.DataFrame({
            "Outcome": [
                "Pass Probability",
                "Fail Probability"
            ],
            "Average Probability": [
                average_pass_probability,
                average_fail_probability
            ]
        })

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Average Pass Probability",
                f"{average_pass_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Average Fail Probability",
                f"{average_fail_probability:.2f}%"
            )

        st.bar_chart(
            probability_df.set_index(
                "Outcome"
            )
        )


        # -------------------------------------------------
        # RECENT ACTIVITY
        # -------------------------------------------------

        st.subheader(
            "Recent Prediction Activity"
        )

        recent_history = history.head(5).copy()

        recent_history["pass_probability"] = (
            recent_history["pass_probability"]
            .apply(lambda x: f"{x:.2f}%")
        )

        recent_history["fail_probability"] = (
            recent_history["fail_probability"]
            .apply(lambda x: f"{x:.2f}%")
        )

        recent_history.columns = [
            "ID",
            "Username",
            "Prediction",
            "Pass Probability",
            "Fail Probability",
            "Prediction Time"
        ]

        st.dataframe(
            recent_history,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No prediction activity is available yet. "
            "Make a prediction from the Prediction System "
            "to populate the dashboard."
        )


    # -----------------------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "Machine Learning Model Performance"
    )

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            "65.82%",
            "67.09%",
            "67.09%"
        ],
        "Precision": [
            "71.67%",
            "70.77%",
            "70.77%"
        ],
        "Recall": [
            "81.13%",
            "86.79%",
            "86.79%"
        ],
        "F1 Score": [
            "76.11%",
            "77.97%",
            "77.97%"
        ],
        "ROC-AUC": [
            "58.85%",
            "62.30%",
            "63.93%"
        ]
    })

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Model metrics are based on the held-out test set. "
        "Prediction activity statistics describe system usage "
        "and should not be interpreted as model accuracy."
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif menu == "Model Information":

    st.title("🤖 Model Information")

    st.subheader(
        "Student Performance Prediction Model"
    )

    st.write(
        "This system uses machine learning to predict "
        "whether a student is likely to Pass or Fail."
    )

    st.divider()

    information = {
        "Dataset": "UCI Student Performance",
        "Number of Records": "395",
        "Problem Type": "Binary Classification",
        "Target": "Pass / Fail",
        "Number of Input Features": "30",
        "Final Model": "Random Forest",
        "Accuracy": "67.09%",
        "F1 Score": "77.97%",
        "ROC-AUC": "63.93%"
    }

    info_df = pd.DataFrame(
        list(information.items()),
        columns=["Item", "Value"]
    )

    st.table(info_df)

    st.divider()

    st.subheader("Model File")

    st.write(
        MODEL_PATH
    )

    if os.path.exists(MODEL_PATH):

        st.success(
            "Model file exists and was loaded successfully."
        )

    else:

        st.error(
            "Model file was not found."
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

elif menu == "Prediction History":

    st.title("📋 Prediction History")

    st.write(
        "Historical prediction records stored in the SQLite database."
    )

    history = get_prediction_history()

    if history.empty:

        st.info(
            "No prediction records are available yet."
        )

    else:

        total_predictions = len(history)

        pass_predictions = (
            history["prediction"] == "Pass"
        ).sum()

        fail_predictions = (
            history["prediction"] == "Fail"
        ).sum()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Predictions",
                total_predictions
            )

        with col2:

            st.metric(
                "Predicted Pass",
                pass_predictions
            )

        with col3:

            st.metric(
                "Predicted Fail",
                fail_predictions
            )

        st.divider()

        display_history = history.copy()

        display_history["pass_probability"] = (
            display_history["pass_probability"]
            .apply(lambda x: f"{x:.2f}%")
        )

        display_history["fail_probability"] = (
            display_history["fail_probability"]
            .apply(lambda x: f"{x:.2f}%")
        )

        display_history.columns = [
            "ID",
            "Username",
            "Prediction",
            "Pass Probability",
            "Fail Probability",
            "Prediction Time"
        ]

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PREDICTION SYSTEM
# =========================================================

elif menu == "Prediction System":

    st.title("🎯 Student Performance Prediction")

    st.write(
        "Enter the student's information below to generate "
        "a Pass or Fail prediction."
    )

    st.divider()


    # -----------------------------------------------------
    # PERSONAL INFORMATION
    # -----------------------------------------------------

    st.subheader("1. Personal Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        school = st.selectbox(
            "School",
            ["GP", "MS"]
        )

        sex = st.selectbox(
            "Sex",
            ["F", "M"]
        )

    with col2:

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=25,
            value=17
        )

        address = st.selectbox(
            "Address",
            ["U", "R"]
        )

    with col3:

        famsize = st.selectbox(
            "Family Size",
            ["LE3", "GT3"]
        )

        Pstatus = st.selectbox(
            "Parent Status",
            ["T", "A"]
        )

    with col4:

        Medu = st.number_input(
            "Mother Education",
            min_value=0,
            max_value=4,
            value=2
        )

        Fedu = st.number_input(
            "Father Education",
            min_value=0,
            max_value=4,
            value=2
        )


    # -----------------------------------------------------
    # FAMILY INFORMATION
    # -----------------------------------------------------

    st.subheader("2. Family Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        Mjob = st.selectbox(
            "Mother Job",
            [
                "teacher",
                "health",
                "services",
                "at_home",
                "other"
            ]
        )

    with col2:

        Fjob = st.selectbox(
            "Father Job",
            [
                "teacher",
                "health",
                "services",
                "at_home",
                "other"
            ]
        )

    with col3:

        reason = st.selectbox(
            "Reason for Choosing School",
            [
                "home",
                "reputation",
                "course",
                "other"
            ]
        )

    with col4:

        guardian = st.selectbox(
            "Guardian",
            [
                "mother",
                "father",
                "other"
            ]
        )


    # -----------------------------------------------------
    # ACADEMIC INFORMATION
    # -----------------------------------------------------

    st.subheader("3. Academic Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        studytime = st.number_input(
            "Study Time",
            min_value=1,
            max_value=4,
            value=2
        )

        failures = st.number_input(
            "Previous Failures",
            min_value=0,
            max_value=4,
            value=0
        )

    with col2:

        traveltime = st.number_input(
            "Travel Time",
            min_value=1,
            max_value=4,
            value=1
        )

        absences = st.number_input(
            "Absences",
            min_value=0,
            max_value=100,
            value=5
        )

    with col3:

        schoolsup = st.selectbox(
            "School Support",
            ["yes", "no"]
        )

        famsup = st.selectbox(
            "Family Support",
            ["yes", "no"]
        )

    with col4:

        paid = st.selectbox(
            "Extra Paid Classes",
            ["yes", "no"]
        )

        higher = st.selectbox(
            "Higher Education Intention",
            ["yes", "no"]
        )


    # -----------------------------------------------------
    # SOCIAL / LIFESTYLE INFORMATION
    # -----------------------------------------------------

    st.subheader("4. Social and Lifestyle Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        activities = st.selectbox(
            "Extra-curricular Activities",
            ["yes", "no"]
        )

        nursery = st.selectbox(
            "Attended Nursery",
            ["yes", "no"]
        )

    with col2:

        internet = st.selectbox(
            "Internet Access",
            ["yes", "no"]
        )

        romantic = st.selectbox(
            "Romantic Relationship",
            ["yes", "no"]
        )

    with col3:

        famrel = st.number_input(
            "Family Relationship Quality",
            min_value=1,
            max_value=5,
            value=4
        )

        freetime = st.number_input(
            "Free Time",
            min_value=1,
            max_value=5,
            value=3
        )

    with col4:

        goout = st.number_input(
            "Going Out",
            min_value=1,
            max_value=5,
            value=3
        )

        health = st.number_input(
            "Health",
            min_value=1,
            max_value=5,
            value=3
        )


    # -----------------------------------------------------
    # LIFESTYLE FACTORS
    # -----------------------------------------------------

    st.subheader("5. Lifestyle Factors")

    col1, col2 = st.columns(2)

    with col1:

        Dalc = st.number_input(
            "Workday Alcohol Consumption",
            min_value=1,
            max_value=5,
            value=1
        )

    with col2:

        Walc = st.number_input(
            "Weekend Alcohol Consumption",
            min_value=1,
            max_value=5,
            value=1
        )


    # -----------------------------------------------------
    # INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "school": [school],
        "sex": [sex],
        "age": [age],
        "address": [address],
        "famsize": [famsize],
        "Pstatus": [Pstatus],
        "Medu": [Medu],
        "Fedu": [Fedu],
        "Mjob": [Mjob],
        "Fjob": [Fjob],
        "reason": [reason],
        "guardian": [guardian],
        "traveltime": [traveltime],
        "studytime": [studytime],
        "failures": [failures],
        "schoolsup": [schoolsup],
        "famsup": [famsup],
        "paid": [paid],
        "activities": [activities],
        "nursery": [nursery],
        "higher": [higher],
        "internet": [internet],
        "romantic": [romantic],
        "famrel": [famrel],
        "freetime": [freetime],
        "goout": [goout],
        "Dalc": [Dalc],
        "Walc": [Walc],
        "health": [health],
        "absences": [absences]

    })


    st.divider()


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Student Performance",
        use_container_width=True
    ):

        try:

            prediction_result = model.predict(
                input_data
            )[0]

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = model.classes_

            probability_dict = dict(
                zip(classes, probabilities)
            )

            pass_probability = (
                probability_dict.get(1, 0) * 100
            )

            fail_probability = (
                probability_dict.get(0, 0) * 100
            )

            if prediction_result == 1:

                prediction_label = "Pass"

            else:

                prediction_label = "Fail"


            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            if prediction_label == "Pass":

                st.success(
                    "Predicted Result: PASS"
                )

                recommendation = (
                    "The model predicts that the student "
                    "is likely to pass. Continue monitoring "
                    "attendance, study habits and academic "
                    "progress."
                )

            else:

                st.error(
                    "Predicted Result: FAIL"
                )

                recommendation = (
                    "The model predicts that the student "
                    "may be at risk of failing. Consider "
                    "additional academic support, closer "
                    "monitoring and early intervention."
                )


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Pass Probability",
                    f"{pass_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Fail Probability",
                    f"{fail_probability:.2f}%"
                )


            # -------------------------------------------------
            # PROBABILITY CHART
            # -------------------------------------------------

            probability_df = pd.DataFrame({
                "Outcome": [
                    "Pass",
                    "Fail"
                ],
                "Probability": [
                    pass_probability,
                    fail_probability
                ]
            })

            st.bar_chart(
                probability_df.set_index(
                    "Outcome"
                )
            )


            # -------------------------------------------------
            # RECOMMENDATION
            # -------------------------------------------------

            st.subheader(
                "Recommendation"
            )

            st.info(
                recommendation
            )


            # -------------------------------------------------
            # SAVE PREDICTION
            # -------------------------------------------------

            save_prediction(
                username=st.session_state.username,
                prediction=prediction_label,
                pass_probability=pass_probability,
                fail_probability=fail_probability
            )

            st.success(
                "Prediction has been saved to the database."
            )


        except Exception as e:

            st.error(
                "An error occurred while making the prediction."
            )

            st.code(
                str(e)
            )
