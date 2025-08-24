import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="bala-ai/tourism_package_purchase_model", filename="best_tourism_package_purchase_model_v1.joblib")
model = joblib.load(model_path)
# Streamlit UI for Tourism package purchase Prediction
st.markdown(
    "<h1 style='white-space: nowrap;'>🏝️ Tourism Package Purchase Prediction App</h1>",
    unsafe_allow_html=True
)
# st.title("Tourism package purchase Prediction App")
st.write("""
This application predicts the likelihood of Tourism package purchased by the customer based on its operational parameters.
Please enter the following details to get a prediction.
""")
# ----------------- User Inputs -----------------

st.subheader("📊 Customer  Data")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, value=10000, step=1000)

with col2:
    OwnCar_option = st.radio("Own Car", ["No", "Yes"], horizontal=True)
    Passport_option = st.radio("Passport", ["No", "Yes"],horizontal=True)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=100, value=1, step=1)
    NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=100, value=5, step=1)
    NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=20, value=0, step=1)
    PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=0.0, max_value=5.0, value=5.0, step=1.0)


st.markdown("---")
st.subheader("📊 Customer Interaction Data")

col3, col4 = st.columns(2)

with col3:
    ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
    DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=1, value=10, step=1)

with col4:
    PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=4, value=3, step=1)
    NumberOfFollowups = st.number_input("Number Of Followups", min_value=1, max_value=10, value=4, step=1)

# ----------------- Preprocessing -----------------
OwnCar = 1 if OwnCar_option == "Yes" else 0
Passport = 1 if Passport_option == "Yes" else 0

# Validation Check
if NumberOfChildrenVisiting > NumberOfPersonVisiting:
    st.error("⚠️ Number of children visiting cannot exceed total number of persons visiting.")

# Assemble user input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'OwnCar': OwnCar,
    'Passport': Passport,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips
}])



if st.button("Predict Purchase"):
    print("input_data--->",input_data)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    result = "Package will be purchased" if prediction == 1 else "Package won't be purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
    st.info(f"Purchase likelihood: {probability:.2%}")
