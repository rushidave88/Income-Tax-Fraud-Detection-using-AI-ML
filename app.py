import streamlit as st
import pandas as pd
import joblib
import requests
from xgboost import XGBClassifier

# Define sandbox client credentials
CLIENT_ID = "16500606a874bffea9aac3337b600561"
CLIENT_SECRET = "cfsk_ma_prod_d24ff27c2b2c6ea400a9ea029aecf6b8_79663b1d"

# Load the label encoders
label_encoder_occupation = joblib.load("label_encoder_occupation.joblib")
label_encoder_marital = joblib.load("label_encoder_marital_status.joblib")
label_encoder_children = joblib.load("label_encoder_children.joblib")

def load_best_model():
    return joblib.load("best_model.joblib")

def predict_income(model, input_data):
    try:
        input_data = input_data.reshape(1, -1)
        return model.predict(input_data)[0]
    except ValueError as e:
        raise ValueError(f"Prediction error: {e}")

def classify_fraud(reported_income, predicted_income):
    def get_tax_slab(income):
        if income <= 300000:
            return 0
        elif 300000 < income <= 600000:
            return 5
        elif 600000 < income <= 900000:
            return 10
        elif 900000 < income <= 1200000:
            return 15
        elif 1200000 < income <= 1500000:
            return 20
        else:
            return 30

    reported_slab = get_tax_slab(reported_income)
    predicted_slab = get_tax_slab(predicted_income)

    return "Fraud" if reported_slab != predicted_slab else "Not Fraud"

def calculate_tax(income):
    if income <= 300000:
        return 0
    elif income <= 600000:
        return 0.05 * (income - 300000)
    elif income <= 900000:
        return 0.1 * (income - 600000) + 0.05 * 300000
    elif income <= 1200000:
        return 0.15 * (income - 900000) + 0.1 * 300000 + 0.05 * 300000
    elif income <= 1500000:
        return 0.20 * (income - 1200000) + 0.15 * 300000 + 0.1 * 300000 + 0.05 * 300000
    else:
        return 0.30 * (income - 1500000) + 0.20 * 300000 + 0.15 * 300000 + 0.1 * 300000 + 0.05 * 300000

def validate_pan_card(pan_card):
    if len(pan_card) != 10 or not (pan_card[:5].isalpha() and pan_card[5:9].isdigit() and pan_card[9].isalpha()):
        raise ValueError("Invalid PAN card format. Please enter in the format AAAAA0000A.")

def validate_aadhar_bank(account_number):
    if not (account_number.isdigit() and len(account_number) == 12):
        raise ValueError("Invalid Aadhar/Bank Account number. It should be 12 digits long.")

def verify_pan_with_cashfree_sync(name, pan_card):
    url = "https://api.cashfree.com/verification/pan"
    headers = {
        "x-client-id": CLIENT_ID,
        "x-client-secret": CLIENT_SECRET,
        "Content-Type": "application/json",
    }
    payload = {"name": name, "pan": pan_card}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def generate_otp_for_aadhar(aadhar_number):
    url = "https://api.cashfree.com/verification/offline-aadhaar/otp"
    headers = {
        "x-client-id": CLIENT_ID,
        "x-client-secret": CLIENT_SECRET,
        "Content-Type": "application/json",
    }
    payload = {"aadhaar_number": aadhar_number}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def verify_aadhar_with_cashfree(otp, aadhar_number, ref_id):
    url = "https://api.cashfree.com/verification/offline-aadhaar/verify"
    headers = {
        "x-client-id": CLIENT_ID,
        "x-client-secret": CLIENT_SECRET,
        "Content-Type": "application/json",
    }
    payload = {"otp": otp, "aadhaar_number": aadhar_number, "ref_id": ref_id}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def main():
    st.title("Fraud Detection App")

    name = st.text_input("Name")
    pan_card = st.text_input("PAN Card")
    aadhar_card = st.text_input("Aadhar Card Number")
    otp = st.text_input("OTP for Aadhar Verification")
    bank_account_no = st.text_input("Bank Account Number")
    ref_id = st.session_state.get('ref_id')

    if st.button("Verify PAN"):
        try:
            validate_pan_card(pan_card)
            response = verify_pan_with_cashfree_sync(name, pan_card)
            st.write(response)
        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"PAN Verification Error: {e}")

    if st.button("Generate OTP for Aadhar"):
        try:
            validate_aadhar_bank(aadhar_card)
            response = generate_otp_for_aadhar(aadhar_card)
            st.session_state['ref_id'] = response.get('ref_id')
            st.write(response)
        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"OTP Generation Error: {e}")

    if st.button("Verify Aadhar"):
        try:
            if not ref_id:
                st.error("Generate OTP first to get ref_id.")
            else:
                response = verify_aadhar_with_cashfree(otp, aadhar_card, ref_id)
                st.write(response)
        except Exception as e:
            st.error(f"Aadhar Verification Error: {e}")

    age = st.slider("Age", 20, 100, 30)
    occupation = st.selectbox("Occupation", ["Salaried", "Self-employed", "Business"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    children = st.selectbox("Children (Yes/No)", ["No", "Yes"])
    reported_income = st.number_input("Reported Income")

    educational_expenses = st.number_input("Educational Expenses") if children == "Yes" else 0
    business_income = st.number_input("Business Income") if occupation == "Business" else 0
    interest_income = st.number_input("Interest Income") if st.selectbox("Do you have interest income? (Yes/No)", ["No", "Yes"]) == "Yes" else 0
    capital_gains = st.number_input("Capital Gains")
    other_income = st.number_input("Other Income")
    healthcare_costs = st.number_input("Healthcare Costs")
    lifestyle_expenditure = st.number_input("Lifestyle Expenditure")
    other_expenses = st.number_input("Other Expenses")
    bank_Debit = st.number_input("Bank_Debited")
    bank_credit=st.number_input("Bank_Credited")
    
    input_data = pd.DataFrame({
       'age': [age],
        'occupation': label_encoder_occupation.transform([occupation])[0],
        'marital_status': label_encoder_marital.transform([marital_status])[0],
        'children': label_encoder_children.transform([children])[0],
        'reported_income': [reported_income],
        'educational_expenses': [educational_expenses],
        'business_income': [business_income],
        'interest_income': [interest_income],
        'capital_gains': [capital_gains],
        'other_income': [other_income],
        'bank_Debit': [bank_Debit],
        ' bank_credit': [bank_credit],
        
        'healthcare_costs': [healthcare_costs],
        'lifestyle_expenditure': [lifestyle_expenditure],
        'other_expenses': [other_expenses],
    })

    if st.button("Predict Income"):
        try:
            best_model = load_best_model()
            predicted_income = predict_income(best_model, input_data.values[0])
            result = classify_fraud(reported_income, predicted_income)
            st.write(f"Prediction: {result}")
            st.write(f"Predicted Income: {predicted_income}")
            st.write(f"Reported Income: {reported_income}")
            st.write(f"Predicted Tax: {calculate_tax(predicted_income)}")
            st.write(f"Reported Tax: {calculate_tax(reported_income)}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()
