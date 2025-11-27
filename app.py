import streamlit as st
import joblib

# Load model
model = joblib.load("final_model.pkl")

st.title("Recommendation System")

# Input fields
user_id = st.number_input("Enter User ID", min_value=1, step=1)
item_id = st.number_input("Enter Item ID", min_value=1, step=1)

if st.button("Predict Rating"):
    try:
        prediction = model.predict(user_id, item_id)
        st.success(f"Predicted Rating: {prediction.est}")
    except Exception as e:
        st.error(f"Error: {e}")

st.write("---")
st.write("Model loaded successfully. Ready to make predictions.")