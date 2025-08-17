import streamlit as st
from model_helper import predict, CLASS_NAMES
from PIL import Image
import io
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="CarShield Insurance - Damage Assessment",
    page_icon="🚗",
    layout="centered", # Can be "wide" for more horizontal space
    initial_sidebar_state="collapsed"
)
# --- Custom CSS for a professional look (similar to an insurance company website) ---
st.markdown(
    """
    <style>
    /* Google Fonts - Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap'); /* Added 800 weight for bolder text */

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #F0FFFF; /* Slightly darker text for better contrast */
    }
    
    .main-header {
        font-size: 2.9em; /* Slightly larger */
        color: #1E90FF; /* Deep Blue for branding */
        text-align: center;
        margin-bottom: 0.5em;
        font-weight: 800; /* Bolder font for main header */
        letter-spacing: -1.5px; /* Tighter letter spacing */
    }

    .sub-header {
        font-size: 1.3em; /* Slightly larger */
        color: #; /* Darker grey for better visibility */
        text-align: center;
        margin-bottom: 2em;
        font-weight: 500; /* Slightly bolder */
    }

    .stButton>button {
        background-color: #007BFF; /* Primary blue for action */
        color: white;
        padding: 0.9em 1.8em; /* Slightly larger padding */
        border-radius: 10px; /* More rounded */
        border: none;
        font-size: 1.2em; /* Slightly larger font */
        font-weight: 700; /* Bolder font for buttons */
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 5px 8px rgba(0, 0, 0, 0.15); /* Stronger shadow */
    }

    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        transform: translateY(-3px); /* More pronounced lift */
    }

    .stFileUploader label {
        color: #004D99;
        font-size: 1.2em; /* Larger font for uploader label */
        font-weight: 600; /* Bolder */
    }

    .stAlert {
        border-radius: 10px;
        font-size: 1.1em; /* Slightly larger */
    }

    .stProgress > div > div > div > div {
        background-color: #28a745; /* Green for success/progress */
    }

    .prediction-card {
        background-color: #f8fbff; /* Very light blue for results, almost white */
        padding: 25px; /* More padding */
        border-radius: 15px; /* More rounded */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* Stronger shadow */
        margin-top: 35px;
        border: 1px solid #bbddee; /* More prominent border */
    }

    .prediction-text {
        font-size: 1.8em; /* Larger */
        font-weight: 700; /* Bolder */
        color: #0069D9; /* A slightly different shade of blue */
        text-align: center;
        margin-bottom: 18px;
    }

    .confidence-text {
        font-size: 1.2em; /* Larger */
        color: #333;
        text-align: center;
        margin-top: 12px;
        font-weight: 600; /* Bolder */
    }

    .probability-bar-container {
        margin-top: 25px;
    }

    .probability-label {
        font-weight: 600; /* Bolder */
        color: #333; /* Darker */
        margin-bottom: 5px;
    }

    .st-emotion-cache-1pxazr7{
        margin-bottom: -35px;
    }

    .logo-container {
        text-align: center;
        margin-bottom: 20px;
    }

    .logo {
        max-width: 200px; /* Adjust logo size */
        height: auto;
        border-radius: 8px; /* Slightly rounded corners for the logo */
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px; /* More padding */
        margin-top: 60px;
        font-size: 1em; /* Slightly larger */
        color: #666; /* Darker grey */
        border-top: 1px solid #dddddd; /* More visible border */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Logo and Header Section ---
st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
# Placeholder logo. You can replace this URL with your actual logo image URL.
# Using a higher resolution placeholder and making it square for better fit.
#st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Fpremium-vector%2Fcar-insurance-automotive-logo-design-illustration_45257887.htm&psig=AOvVaw1qoNrILqyuqHwC0C0xyHPX&ust=1755532686400000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCIiQn4Sbko8DFQAAAAAdAAAAABAE",
         #caption="CarShield Insurance Logo",
         #use_column_width=True,
         #width=150, # Set a fixed width for consistent logo size
         #output_format="PNG", # Ensure consistent format
         
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>CarShield Insurance</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your trusted partner for quick and accurate car damage assessments.</p>", unsafe_allow_html=True)

st.write("---")

# --- Main Content: Damage Assessment Tool ---
st.markdown("## 🛠️ Damage Assessment Tool", unsafe_allow_html=True)
st.write("Upload an image of your car's damage below, and our AI will help identify the type of damage instantly.")

uploaded_file = st.file_uploader(
    "Choose an image file (JPG, PNG)",
    type=['jpg', 'jpeg', 'png'],
    help="Please upload a clear image of the car damage."
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded file temporarily to pass to the prediction model
    image_path = "temp_uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("---")

    # Predict button
    if st.button("Analyze Damage", key="analyze_button"):
        with st.spinner("Analyzing image... Please wait."):
            # Perform prediction
            predicted_class, probabilities = predict(image_path, return_probabilities=True)

            # Check for prediction errors
            if predicted_class.startswith("Error"):
                st.error(f"Prediction Error: {predicted_class}")
            else:
                # Display results in a styled card
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown(f"<p class='prediction-text'>Predicted Damage: <strong>{predicted_class}</strong></p>", unsafe_allow_html=True)

                # Find the confidence for the predicted class
                predicted_confidence = probabilities[CLASS_NAMES.index(predicted_class)] * 100
                st.markdown(f"<p class='confidence-text'>Confidence: <strong>{predicted_confidence:.2f}%</strong></p>", unsafe_allow_html=True)

                st.markdown("<div class='probability-bar-container'>", unsafe_allow_html=True)
                st.write("---")
                st.subheader("Probability Distribution")
                # Display probabilities for all classes using Streamlit's progress bar
                for i, class_name in enumerate(CLASS_NAMES):
                    prob_percentage = probabilities[i] * 100
                    st.markdown(f"<p class='probability-label'>{class_name}</p>", unsafe_allow_html=True)
                    st.progress(float(prob_percentage / 100)) # st.progress expects float between 0.0 and 1.0
                    st.markdown(f"<p style='text-align: right; margin-top: -30px; margin-bottom: 20px;'>{prob_percentage:.2f}%</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Clean up the temporary file
    if os.path.exists(image_path):
        os.remove(image_path)

st.write("---")

# --- Information Section ---
st.markdown("## ℹ️ How it Works", unsafe_allow_html=True)
st.write(
    """
    Our advanced AI model, trained on a comprehensive dataset of car images,
    can quickly classify various types of vehicle damage. Simply upload an image,
    and our system will provide an immediate assessment. This helps streamline
    the claims process and gives you a preliminary understanding of the damage.
    """
)

st.markdown("## 📞 Contact Us", unsafe_allow_html=True)
st.write(
    """
    Have questions or need to file a claim?
    Reach out to our customer support team:
    * **Phone:** +49 15563274805
    * **Email:** sudiptapriyam55@gmail.com
    * **Address:** Siegen,North Rhine Westaphia, DEU 57072
    """
)

# --- Footer ---
st.markdown("<div class='footer'>© 2024 CarShield Insurance. All rights reserved.</div>", unsafe_allow_html=True)
