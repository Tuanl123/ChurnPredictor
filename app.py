import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the model architecture 
class Classification(nn.Module):
    def __init__(self, inputs_count, nodes_count, classes):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.001)

    # Define 5 layers
        self.hl1 = nn.Linear(inputs_count, nodes_count)
        self.hl2 = nn.Linear(nodes_count, nodes_count)
        self.hl3 = nn.Linear(nodes_count, nodes_count)
        self.hl4 = nn.Linear(nodes_count, nodes_count)
        self.hl5 = nn.Linear(nodes_count, classes)

        self.double()
    
    # Pass data through all layers
    def forward(self, inputs):
        outputs = self.activation(self.hl1(inputs))
        outputs = self.activation(self.hl2(outputs))
        outputs = self.activation(self.hl3(outputs))
        outputs = self.activation(self.hl4(outputs))
        outputs = self.activation(self.hl5(outputs))
        return outputs

# Function to load model
@st.cache_resource
# Keep model in streamlit cache
def load_model():
    inputs_size = 30  # Number of features after one-hot encoding
    nodes_size = 32 # Number of nodes per hidden layer 
    classes = 2 # Number of output classes
    
    model = Classification(inputs_count=inputs_size, nodes_count=nodes_size, classes=classes)
    
    # Check if CUDA is available, otherwise use CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        # Load the model state dict
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.to(device)
        model.eval()
        return model, device
    # Error for model, returns none
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

# Function to preprocess input data
def preprocess_input(data):
    # Convert input to DataFrame
    df = pd.DataFrame([data])
    
    # Get numerical features (before onehot encoding)
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    # Z-score normalization parameters,dummy numbers
    # Get this from training data (remember to do it, please please)
    normalization_params = {
        "tenure": {"mean": 32.4, "std": 24.6},
        "MonthlyCharges": {"mean": 64.8, "std": 30.1},
        "TotalCharges": {"mean": 2280.0, "std": 2266.8}
    }
    
    # Apply z-score normalization (value - mean)/std
    for feature in numerical_features:
        mean = normalization_params[feature]["mean"]
        std = normalization_params[feature]["std"]
        df[feature] = (df[feature] - mean) / std
    
    # One-hot encode categorical features
    categorical_features = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]
    
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Define columns that model expects. MAKE SURE ALL EXPECTED COLUMNS ARE PRESENT
    expected_columns = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'gender_Male', 'Partner_Yes', 'Dependents_Yes',
        'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year',
        'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)', 
        'PaymentMethod_Electronic check', 
        'PaymentMethod_Mailed check',
        'PaymentMethod_Bank transfer (automatic)'
    ]
    
    # Add missing columns with default value 0 (for things like No internet)
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Select only the expected columns in the correct order
    df_final = df_encoded[expected_columns]
    
    return df_final

# Function to make prediction
def predict_churn(model, device, preprocessed_data):
    # Convert dataFrame to tensor
    input_tensor = torch.tensor(preprocessed_data.values, dtype=torch.float64).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    # Extract probability and prediction
    churn_prob = probabilities[0][1].item()
    is_churn = bool(predicted.item())
    
    return is_churn, churn_prob

# Main UI
def main():
    st.title("Customer Churn Prediction")
    
    # Load model, error handling
    model, device = load_model()
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return
    
    st.write("""
    ## Predict if a customer will churn :D
    """)
    
    # Create a two-column layout
    col1, col2 = st.columns(2)
    
    # Customer information inputs
    with col1:
        st.subheader("Customer Profile")
        
        gender = st.radio("Gender", ["Female", "Male"])
        partner = st.radio("Partner", ["No", "Yes"])
        dependents = st.radio("Dependents", ["No", "Yes"])
        
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        phone_service = st.radio("Phone Service", ["No", "Yes"])
        
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
            disabled=(phone_service == "No"),
            index=0 if phone_service == "Yes" else 2
        )
        
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
    with col2:
        st.subheader("Services & Billing")
        
        # Depends on internet service
        has_internet = internet_service != "No"
        internet_options = ["Yes", "No", "No internet service"]
        default_internet_option = 1 if has_internet else 2  # "No" if has internet, "No internet service" if not
        
        online_security = st.selectbox(
            "Online Security",
            internet_options,
            disabled=not has_internet,
            index=default_internet_option
        )
        
        online_backup = st.selectbox(
            "Online Backup",
            internet_options,
            disabled=not has_internet,
            index=default_internet_option
        )
        
        device_protection = st.selectbox(
            "Device Protection",
            internet_options,
            disabled=not has_internet,
            index=default_internet_option
        )
        
        tech_support = st.selectbox(
            "Tech Support",
            internet_options,
            disabled=not has_internet,
            index=default_internet_option
        )
        
        streaming_tv = st.selectbox(
            "Streaming TV",
            internet_options,
            disabled=not has_internet,
            index=default_internet_option
        )
        
        streaming_movies = st.selectbox(
            "Streaming Movies",
            internet_options,
            disabled=not has_internet,
            index=default_internet_option
        )
    
    # Create a new row 
    col1, col2 = st.columns(2)
    
    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        
        paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
        
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
    
    with col2:
        st.subheader("Financial Details")
        
        monthly_charges = st.slider(
            "Monthly Charges ($)",
            0.0, 200.0, 65.0, 0.1
        )
        
        # TotalCharges can be calculated as tenure * MonthlyCharges for simplicity
        default_total = round(tenure * monthly_charges, 2)
        use_calculated_total = st.checkbox("Calculate Total Charges based on Tenure and Monthly Charges", value=True)
        
        if use_calculated_total:
            total_charges = default_total
            st.info(f"Total Charges: ${total_charges:.2f}")
        else:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=default_total,
                step=10.0
            )
    
    # Create a dictionary with all input data
    input_data = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    # Button to make prediction
    if st.button("Predict Churn"):
        with st.spinner("Predicting..."):
            # Preprocess input data
            preprocessed_data = preprocess_input(input_data)
            
            # Make prediction
            is_churn, churn_probability = predict_churn(model, device, preprocessed_data)
            
            # Display prediction
            st.header("Prediction Results")
            
            # Create a gauge-like display for the churn probability
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if is_churn:
                    st.error("### Customer is likely to churn!")
                else:
                    st.success("### Customer is likely to stay!")
            
            with col2:
                # Create a custom progress bar for probability
                churn_percentage = int(churn_probability * 100)
                
                # Use different colors based on probability
                if churn_percentage < 30:
                    bar_color = "green"
                elif churn_percentage < 70:
                    bar_color = "orange"
                else:
                    bar_color = "red"
                
                st.markdown(f"### Churn Probability: {churn_percentage}%")
                st.progress(churn_probability)
                
                # Add interpretation
                if churn_percentage < 30:
                    st.success("Low risk of churn")
                elif churn_percentage < 70:
                    st.warning("Moderate risk of churn")
                else:
                    st.error("High risk of churn")
            
            # Display insights based on the input
            st.subheader("Customer Insights")
            
            insights = []
            
            # Contract type
            if contract == "Month-to-month":
                insights.append("Month-to-month contracts may have higher churn rates")
            elif contract in ["One year", "Two year"]:
                insights.append("Longer contracts typically have lower churn rates")
            
            # Tenure
            if tenure < 12:
                insights.append("New customers (less than 1 year) tend to churn more frequently")
            elif tenure > 24:
                insights.append("Long-term customers (over 2 years) typically have higher loyalty")
            
            # Fiber optic service
            if internet_service == "Fiber optic":
                insights.append("Fiber optic customers often have higher churn rates despite premium service")
            
            # Payment method
            if payment_method == "Electronic check":
                insights.append("Electronic check payment method may be associated with higher churn")
            
            # Additional services
            services = [online_security, online_backup, device_protection, tech_support]
            yes_count = services.count("Yes")
            if yes_count == 0 and has_internet:
                insights.append("No additional services may indicate lower engagement and higher churn risk")
            elif yes_count >= 3:
                insights.append("Multiple additional services typically indicate higher customer engagement")
            
            # Display insights
            for insight in insights:
                st.info(insight)
            
            # Retention suggestions
            if is_churn:
                st.subheader("Retention Suggestions")
                
                suggestions = []
                
                if contract == "Month-to-month":
                    suggestions.append("Offer discounted long-term contract options")
                
                if payment_method == "Electronic check":
                    suggestions.append("Suggest automatic payment methods for convenience")
                
                if has_internet and yes_count < 2:
                    suggestions.append("Promote service bundles with discounted pricing")
                
                if tenure < 12:
                    suggestions.append("Provide special offers for new customer loyalty")
                
                if monthly_charges > 80:
                    suggestions.append("Consider a customized pricing plan review")
                
                # Display suggestions
                for suggestion in suggestions:
                    st.success(suggestion)

if __name__ == "__main__":
    main()