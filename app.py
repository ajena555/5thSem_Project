import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
model = pickle.load(open('voting.pkl', 'rb'))

# Set the page layout
st.set_page_config(page_title="ASDInsight", page_icon="🔍", layout="wide")

# Sidebar for navigation with a Selectbox
st.sidebar.title("Navigation")
st.sidebar.markdown("<h2 style='color:#FFA500;'>Choose a Page</h2>", unsafe_allow_html=True)  # Custom color
page = st.sidebar.selectbox("", ["Home", "About", "Prediction"], index=0)

# Home page
if page == "Home":
    st.title("Welcome to ASDInsight")
    st.image("static/images/2", use_column_width=True)  # Example image link
    st.subheader("Your personalized Autism Spectrum Disorder (ASD) prediction tool.")
    st.markdown("""
    ### Features:
    - **Accurate ASD predictions** using state-of-the-art machine learning models.
    - **User-friendly interface** to input child symptoms and get predictions in real-time.
    
    ASD IntelliPredict provides caregivers and medical professionals with insights based on symptoms, medical history, and behavioral factors.
    
    Navigate through the app using the sidebar and try our **Prediction** tool to check the likelihood of Autism Spectrum Disorder.
    """)
    
# About page
elif page == "About":
    st.title("About ASD IntelliPredict")
    st.image("static/images/autism-page-img-1.png")
    st.markdown("""
    ### Our Mission:
    ASD IntelliPredict is designed to provide early detection of Autism Spectrum Disorder (ASD) based on common symptoms and behavioral patterns.
    
    - **Backed by Data**: Uses real-world datasets and advanced decision trees to generate accurate predictions.
    - **Supporting Parents & Caregivers**: We aim to help parents understand their child's needs early and seek medical intervention where necessary.
    
    #### How It Works:
    - Input a set of symptoms and related medical factors.
    - The model predicts the likelihood of ASD based on the input data.
    - The tool also provides the probability percentage, helping users make informed decisions.
    
    We do not replace medical professionals, but we aim to be a supportive tool for ASD awareness and early detection.
    """)

# Prediction page
elif page == "Prediction":
    st.title("ASD Prediction Tool")
    
    # Grouped input sections
    st.header("Autism Spectrum Quotient")
    a1 = st.selectbox('Does your child show interest in repetitive patterns?', ["Yes", "No"])
    a2 = st.selectbox('Does your child avoid social interactions?', ["Yes", "No"])
    a3 = st.selectbox('Does your child have difficulty understanding social cues?', ["Yes", "No"])
    a4 = st.selectbox('Is your child sensitive to loud noises?', ["Yes", "No"])
    a5 = st.selectbox('Does your child exhibit repetitive movements?', ["Yes", "No"])
    a6 = st.selectbox('Does your child avoid eye contact?', ["Yes", "No"])
    a7 = st.selectbox('Does your child prefer routines?', ["Yes", "No"])
    a8 = st.selectbox('Does your child have difficulty understanding emotions?', ["Yes", "No"])
    a9 = st.selectbox('Is your child overly focused on specific interests?', ["Yes", "No"])
    a10 = st.number_input('Autism Spectrum Quotient Score (1-10)', min_value=1, max_value=10)

    st.header("Medical History")
    social_scale = st.number_input('Social Responsiveness Scale Score', min_value=0, max_value=100)
    speech_delay = st.selectbox('Speech Delay or Language Disorder', ["Yes", "No"])
    learning_disorder = st.selectbox('Learning Disorder', ["Yes", "No"])
    genetic_disorders = st.selectbox('Any Genetic Disorders?', ["Yes", "No"])
    depression = st.selectbox('Depression or Mood Disorders?', ["Yes", "No"])
    developmental_delay = st.selectbox('Global Developmental Delay or Intellectual Disability', ["Yes", "No"])
    social_issues = st.selectbox('Any Social/Behavioural Issues?', ["Yes", "No"])
    autism_rating = st.number_input('Childhood Autism Rating Scale', min_value=0.0, max_value=60.0)

    st.header("Additional Information")
    anxiety = st.selectbox('Anxiety Disorder', ["Yes", "No"])
    sex = st.selectbox('Sex', ['Male', 'Female'])
    jaundice = st.selectbox('History of Jaundice at Birth', ["Yes", "No"])
    family_asd = st.selectbox('Family Member with ASD', ["Yes", "No"])

    # Convert Yes/No inputs to binary values
    def convert_yes_no(value):
        return 1 if value == "Yes" else 0

    # Collect input data
    input_data = [
        convert_yes_no(a1), convert_yes_no(a2), convert_yes_no(a3), convert_yes_no(a4), convert_yes_no(a5),
        convert_yes_no(a6), convert_yes_no(a7), convert_yes_no(a8), convert_yes_no(a9), a10,
        social_scale, convert_yes_no(speech_delay), convert_yes_no(learning_disorder), convert_yes_no(genetic_disorders),
        convert_yes_no(depression), convert_yes_no(developmental_delay), convert_yes_no(social_issues), autism_rating,
        convert_yes_no(anxiety), 1 if sex == 'Male' else 0, convert_yes_no(jaundice), convert_yes_no(family_asd)
    ]

    # Convert input data to NumPy array and reshape it
    input_data = np.array(input_data).reshape(1, -1)

    # When user clicks "Predict"
    if st.button('Predict'):
        prediction_proba = model.predict_proba(input_data)[0]  # Get the prediction probability

        # Display the probability of having ASD
        st.subheader('Prediction Result:')
        st.write(f"Probability of having ASD: {prediction_proba[1]*100:.2f}%")
        if prediction_proba[1] > 0.5:
            st.error('High Risk of ASD. For early insights consult to an expert without any delay.')
        else:
            st.success('Low Risk of ASD, Still consult to an expert regarding this.')
