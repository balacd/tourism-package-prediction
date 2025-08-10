import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="bala-ai/tourism_package_purchase_model", filename="best_tourism_package_purchase_model_v1.joblib")
model = joblib.load(model_path)

columns_path = hf_hub_download(
    repo_id="bala-ai/tourism_package_purchase_model",
    filename="model_columns.joblib"
)
model_columns = joblib.load(columns_path)

print(model_columns)


# Streamlit UI for Tourism package purchase Prediction
st.title("Tourism package purchase Prediction App")
st.write("""
This application predicts the likelihood of Tourism package purchased by the customer based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input



Age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)


NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=100, value=1, step=1)


NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=100, value=5, step=1)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=20, value=0, step=1)

PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=0.0, max_value=5.0, value=5.0, step=1.0)


MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, value=10000, step=1000)


owncar_option = st.radio("Own Car", ["No", "Yes"])
Passport_option = st.radio("Passport", ["No", "Yes"])




CityTier = st.selectbox("CityTier", [1,2,3])


TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])

Gender = st.selectbox("Gender", ["Male", "Female"])


MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Divorced", "Married", "Unmarried"])

Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
# ---------customer interaction data------------------
ProductPitched = st.selectbox("ProductPitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])

DurationOfPitch = st.number_input("DurationOfPitch", min_value=1, value=10, step=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=4, value=3, step=1)

NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=10, value=4, step=1)


# Convert Yes/No to binary
OwnCar = 1 if owncar_option == "Yes" else 0
Passport = 1 if Passport_option == "Yes" else 0

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


# One-hot encode the input row
input_encoded = pd.get_dummies(input_data, drop_first=True)

# Reindex to match training columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)



if st.button("Predict Purchase"):
    prediction = model.predict(input_encoded)[0]
    result = "Package will be purchased" if prediction == 1 else "Package wont be purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
