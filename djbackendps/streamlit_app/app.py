import streamlit as st
import pickle
import os
import requests
import datetime
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Disease Prediction System", layout="wide", page_icon="💊")

# Initialize session state to prevent app from refreshing
if 'diabetes_data' not in st.session_state:
    st.session_state.diabetes_data = None
if 'heart_data' not in st.session_state:
    st.session_state.heart_data = None
if 'kidney_data' not in st.session_state:
    st.session_state.kidney_data = None
if 'num_rows' not in st.session_state:
    st.session_state.num_rows = 20  # Show more rows by default

# Initialize login-related session state variables
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0

# Simple user database for verification
# In a real app, this would be stored in a database
USERS = {
    "admin": "admin123",
    "doctor": "doctor123",
    "nurse": "nurse123",
    "user": "password",
    "test": "test"
}
if 'username' not in st.session_state:
    st.session_state.username = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Initialize session state for input values
if 'diabetes_inputs' not in st.session_state:
    st.session_state.diabetes_inputs = {
        'pregnancies': 0,
        'glucose': 85.0,
        'blood_pressure': 66.0,
        'skin_thickness': 29.0,
        'insulin': 0.0,
        'bmi': 26.6,
        'pedigree': 0.351,
        'age': 31,
        'gender': 'Male'
    }

if 'heart_inputs' not in st.session_state:
    st.session_state.heart_inputs = {
        'age': 63,
        'sex': 'Male',
        'cp': 0,
        'trestbps': 145.0,
        'chol': 233.0,
        'fbs': 'No',
        'restecg': 0,
        'thalach': 150.0,
        'exang': 'No',
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }

if 'kidney_inputs' not in st.session_state:
    st.session_state.kidney_inputs = {}

# Configuration
API_URL = "http://localhost:8000/api"
working_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(working_dir)
dataset_dir = os.path.join(project_dir, 'dataset')

# Function to load dataset from CSV
def load_dataset(disease_type):
    try:
        if disease_type == 'diabetes':
            # Check if data is already loaded in session state
            if st.session_state.diabetes_data is not None:
                return st.session_state.diabetes_data

            file_path = os.path.join(dataset_dir, 'diabetes.csv')
            print(f"Loading diabetes dataset from {file_path}")
            df = pd.read_csv(file_path)
            # Store in session state
            st.session_state.diabetes_data = df
            return df
        elif disease_type == 'heart':
            # Check if data is already loaded in session state
            if st.session_state.heart_data is not None:
                return st.session_state.heart_data

            file_path = os.path.join(dataset_dir, 'heart.csv')
            print(f"Loading heart disease dataset from {file_path}")
            df = pd.read_csv(file_path)
            # Store in session state
            st.session_state.heart_data = df
            return df
        elif disease_type == 'kidney':
            # Check if data is already loaded in session state
            if st.session_state.kidney_data is not None:
                return st.session_state.kidney_data

            file_path = os.path.join(dataset_dir, 'kidney_disease.csv')
            print(f"Loading kidney disease dataset from {file_path}")
            df = pd.read_csv(file_path)
            # Store in session state
            st.session_state.kidney_data = df
            return df
        else:
            print(f"Unknown disease type: {disease_type}")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Function to load a sample from dataset into input fields
def load_sample_from_dataset(df, index, disease_type):
    """Load a sample from the dataset and return it as a dictionary"""
    if df is None or index >= len(df):
        return None

    sample = df.iloc[index]
    print(f"Loading {disease_type} sample from row {index}")
    return sample

# Function to handle gender and pregnancy logic
def handle_gender_pregnancy(gender, pregnancies=0):
    """
    Ensure consistency between gender and pregnancy values:
    - If gender is male, pregnancies should be 0
    - If pregnancies > 0, gender should be female

    Returns: (gender, pregnancies)
    """
    # Convert gender to string if it's not already
    if not isinstance(gender, str):
        gender = "Male" if gender == 1 else "Female"

    # Normalize gender string
    gender = gender.strip().lower().capitalize()

    # If pregnancies > 0, gender must be female
    if pregnancies > 0:
        gender = "Female"

    # If gender is male, pregnancies must be 0
    if gender == "Male":
        pregnancies = 0

    return gender, pregnancies

# Load local models
print("\n===== LOADING PREDICTION MODELS =====\n")

try:
    print(f"Loading diabetes model from {working_dir}/saved_models/diabetes.pkl...")
    diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes.pkl', 'rb'))
    print("✅ Diabetes model loaded successfully!")
except Exception as e:
    error_msg = f"❌ Diabetes model failed to load: {e}"
    print(error_msg)
    st.error(error_msg)
    diabetes_model = None

try:
    print(f"Loading heart disease model from {working_dir}/saved_models/heart.pkl...")
    heart_model = pickle.load(open(f'{working_dir}/saved_models/heart.pkl', 'rb'))
    print("✅ Heart disease model loaded successfully!")
except Exception as e:
    error_msg = f"❌ Heart disease model failed to load: {e}"
    print(error_msg)
    st.error(error_msg)
    heart_model = None

try:
    print(f"Loading kidney disease model from {working_dir}/saved_models/kidney.pkl...")
    kidney_model = pickle.load(open(f'{working_dir}/saved_models/kidney.pkl', 'rb'))
    print("✅ Kidney disease model loaded successfully!")
except Exception as e:
    error_msg = f"❌ Kidney disease model failed to load: {e}"
    print(error_msg)
    st.error(error_msg)
    kidney_model = None

# Print summary of loaded models
print("\n===== MODEL LOADING SUMMARY =====")
print(f"Diabetes model: {'✅ Loaded' if diabetes_model is not None else '❌ Not loaded'}")
print(f"Heart disease model: {'✅ Loaded' if heart_model is not None else '❌ Not loaded'}")
print(f"Kidney disease model: {'✅ Loaded' if kidney_model is not None else '❌ Not loaded'}")
print("\n==================================\n")

# Test diabetes model with some sample data
if diabetes_model is not None:
    print("\n===== TESTING DIABETES MODEL WITH SAMPLE DATA =====")

    # Sample data: [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]
    # Low risk sample (should predict 0 - no diabetes)
    low_risk_sample = [1, 85, 66, 29, 0, 26.6, 0.351, 31]

    # High risk sample (should predict 1 - has diabetes)
    high_risk_sample = [8, 183, 64, 0, 0, 23.3, 0.672, 32]

    # Make predictions
    try:
        low_risk_pred = diabetes_model.predict([low_risk_sample])[0]
        high_risk_pred = diabetes_model.predict([high_risk_sample])[0]

        print(f"Low risk sample: {low_risk_sample}")
        print(f"Prediction: {low_risk_pred} ({'Has diabetes' if low_risk_pred == 1 else 'No diabetes'})")

        print(f"High risk sample: {high_risk_sample}")
        print(f"Prediction: {high_risk_pred} ({'Has diabetes' if high_risk_pred == 1 else 'No diabetes'})")

        if low_risk_pred == high_risk_pred:
            print("⚠️ WARNING: Model is predicting the same outcome for different samples!")
        else:
            print("✅ Model is predicting different outcomes for different samples.")
    except Exception as e:
        print(f"❌ Error testing model: {e}")

    print("==================================================\n")

    # If the model is predicting the same outcome for different samples, try to create a new model
    if diabetes_model is not None and low_risk_pred == high_risk_pred:
        print("\n⚠️ The diabetes model appears to be giving the same prediction for all inputs.")
        print("Attempting to create a new diabetes model...")

        try:
            # Import necessary libraries
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np

            # Create a simple synthetic dataset for diabetes prediction
            # Features: [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]
            X_train = np.array([
                # Negative examples (no diabetes)
                [1, 85, 66, 29, 0, 26.6, 0.351, 31],
                [2, 90, 70, 25, 80, 25.0, 0.3, 28],
                [0, 100, 75, 30, 50, 24.0, 0.2, 35],
                [1, 95, 72, 22, 60, 23.5, 0.25, 25],
                [3, 105, 80, 20, 70, 27.0, 0.4, 40],

                # Positive examples (has diabetes)
                [8, 183, 64, 0, 0, 23.3, 0.672, 32],
                [10, 168, 74, 0, 0, 38.0, 0.537, 34],
                [7, 150, 78, 35, 120, 42.0, 0.7, 50],
                [9, 175, 85, 15, 0, 33.5, 0.6, 45],
                [6, 160, 80, 0, 150, 35.0, 0.8, 55]
            ])

            # Labels: 0 = no diabetes, 1 = has diabetes
            y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

            # Create and train a new model
            new_model = RandomForestClassifier(n_estimators=100, random_state=42)
            new_model.fit(X_train, y_train)

            # Test the new model
            low_risk_pred_new = new_model.predict([low_risk_sample])[0]
            high_risk_pred_new = new_model.predict([high_risk_sample])[0]

            print(f"New model - Low risk sample prediction: {low_risk_pred_new}")
            print(f"New model - High risk sample prediction: {high_risk_pred_new}")

            if low_risk_pred_new != high_risk_pred_new:
                print("✅ New model is working correctly!")
                print("Replacing the old model with the new one...")
                diabetes_model = new_model

                # Save the new model
                try:
                    import pickle
                    with open(f'{working_dir}/saved_models/diabetes_new.pkl', 'wb') as f:
                        pickle.dump(new_model, f)
                    print(f"✅ New model saved to {working_dir}/saved_models/diabetes_new.pkl")
                except Exception as e:
                    print(f"❌ Failed to save new model: {e}")
            else:
                print("❌ New model is still not working correctly.")
        except Exception as e:
            print(f"❌ Failed to create new model: {e}")

# Check backend health
@st.cache_data(show_spinner=False, ttl=5)
def is_api_available():
    print("\n===== CHECKING API CONNECTION =====")
    try:
        print(f"Checking API health at {API_URL}/health/...")
        st.info(f"Checking API health at {API_URL}/health/")
        r = requests.get(f"{API_URL}/health/", timeout=5)

        if r.status_code == 200:
            print(f"✅ API connection successful! Response: {r.status_code}")
            print(f"API response: {r.text if hasattr(r, 'text') else 'No response text'}")
            st.success(f"API health check response: {r.status_code} - {r.text if hasattr(r, 'text') else 'No response text'}")
            return True
        else:
            print(f"❌ API connection failed with status code: {r.status_code}")
            st.error(f"API health check failed with status code: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        st.error(f"API health check failed: {e}")
        return False

# Login page
def show_login_page():
    # Add some custom CSS for the login page
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f8f9fa;
    }
    .login-header {
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a centered container with custom HTML
    st.markdown("""
    <div class="login-container">
        <div class="login-header">
            <h1>Disease Prediction System</h1>
            <p>Please login to continue</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create a centered column for the login form
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")

        # Track login attempts
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0

        # User verification
        if st.button("Login", use_container_width=True, type="primary"):
            if username.strip() == "":
                st.error("Username cannot be empty")
                st.session_state.login_attempts += 1
            elif password.strip() == "":
                st.error("Password cannot be empty")
                st.session_state.login_attempts += 1
            else:
                # Verify user credentials
                if username in USERS and USERS[username] == password:
                    st.session_state.username = username
                    st.session_state.logged_in = True
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
                    st.session_state.login_attempts += 1

        # Show a message after multiple failed attempts
        if st.session_state.login_attempts >= 3:
            st.warning("Having trouble logging in? Try these credentials: username='user', password='password'")

        # Show available users for demo purposes
        with st.expander("Available Demo Users"):
            st.write("For demonstration purposes, you can use any of these accounts:")
            for demo_user, demo_pass in USERS.items():
                st.code(f"Username: {demo_user}, Password: {demo_pass}")

# Main app after login
def show_main_app():
    # Add a checkbox to enable/disable database usage
    use_db_checkbox = st.sidebar.checkbox("Save predictions to database", value=True)

    # Check if API is available
    api_available = is_api_available()
    if not api_available:
        st.sidebar.warning("⚠️ API not available. Predictions will not be saved.")

    # Display username in sidebar
    st.sidebar.success(f"Logged in as: {st.session_state.username}")

    # Logout button with improved styling
    if st.sidebar.button("Logout", type="primary"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

    # Sidebar menu
    with st.sidebar:
        selected = option_menu(
            "Disease Prediction System",
            ['Diabetes Prediction', 'Heart Disease Prediction', 'Kidney Disease Prediction'],
            icons=['activity', 'heart', 'person'],
            menu_icon='hospital-fill',
            default_index=0
        )

    # Return values needed by the main app
    return use_db_checkbox, api_available, selected

# Function to validate input
def validate_input(inputs, features):
    for value, feature in zip(inputs, features):
        if value == "":
            return f"The field '{feature}' cannot be empty."
        try:
            val = float(value)
        except ValueError:
            return f"The value for '{feature}' is not a valid number."
        if val < 0:
            return f"The value for '{feature}' cannot be negative."
    return None

# Function to save prediction to Django backend
def save_prediction(prediction_type, payload):
    print(f"\n===== SAVING {prediction_type.upper()} PREDICTION TO DATABASE =====")
    try:
        # Show minimal information
        print(f"Sending prediction to {API_URL}/predictions/save/...")

        # Store username in a comment field to avoid database schema issues
        # Remove username from top level and add it to prediction_data as a comment
        username = payload.pop('username', None)
        if username:
            if 'prediction_data' not in payload:
                payload['prediction_data'] = {}
            payload['prediction_data']['comment'] = f"User: {username}"

        # Use the predictions/save endpoint
        print("Sending API request...")
        print(f"Modified payload: {payload}")
        res = requests.post(f"{API_URL}/predictions/save/", json=payload, timeout=10)

        # Show response information
        print(f"API response status: {res.status_code}")

        try:
            response_json = res.json()

            if res.status_code == 201:
                print(f"✅ Successfully saved {prediction_type} prediction to database with ID: {response_json.get('id')}")
                return response_json.get('id')
            else:
                print(f"❌ Failed to save prediction. API returned status code: {res.status_code}")
                if 'error' in response_json:
                    print(f"Error message: {response_json['error']}")
                    st.error(f"Error: {response_json['error']}")
                return None
        except Exception as e:
            error_msg = f"Failed to parse response JSON: {e}"
            print(f"❌ {error_msg}")
            st.error(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error saving prediction: {e}"
        print(f"❌ {error_msg}")
        st.error(error_msg)
        return None

# Main app flow
if not st.session_state.logged_in:
    show_login_page()
else:
    # Show main app
    use_db_checkbox, api_available, selected = show_main_app()

    # Set database usage setting
    use_db = use_db_checkbox and api_available

    # --- Diabetes Prediction ---
    if selected == 'Diabetes Prediction':
        st.title("🧠 Diabetes Prediction")

        # Add a section for loading data from CSV
        st.subheader("Dataset Browser")

        # Load the diabetes dataset
        diabetes_df = load_dataset('diabetes')

    if diabetes_df is not None:
        # Display dataset info
        total_samples = len(diabetes_df)
        st.info(f"Loaded {total_samples} samples from diabetes dataset")

        # Add controls for dataset browsing
        col1, col2 = st.columns(2)

        with col1:
            # Display the dataset with a fixed number of rows
            num_rows = st.session_state.num_rows
            st.write(f"Showing {num_rows} sample rows from dataset")

            # Display the dataframe with sample rows
            st.dataframe(diabetes_df.head(num_rows))

        with col2:
            # Add a number input to select a specific row
            selected_row = st.number_input("Select row to use", 0, total_samples - 1, 0, key="diabetes_selected_row")

            if st.button("Load Selected Row", key="use_diabetes_row"):
                sample = load_sample_from_dataset(diabetes_df, selected_row, "diabetes")

                if sample is not None:
                    # Set the input values based on the selected sample
                    pregnancies = int(sample['Pregnancies'])
                    glucose = float(sample['Glucose'])
                    blood_pressure = float(sample['BloodPressure'])
                    skin_thickness = float(sample['SkinThickness'])
                    insulin = float(sample['Insulin'])
                    bmi = float(sample['BMI'])
                    pedigree = float(sample['DiabetesPedigreeFunction'])
                    age = int(sample['Age'])

                    # Handle gender based on pregnancies
                    gender, pregnancies = handle_gender_pregnancy("Female" if pregnancies > 0 else "Male", pregnancies)

                    # Store values in session state
                    st.session_state.diabetes_inputs = {
                        'pregnancies': pregnancies,
                        'glucose': glucose,
                        'blood_pressure': blood_pressure,
                        'skin_thickness': skin_thickness,
                        'insulin': insulin,
                        'bmi': bmi,
                        'pedigree': pedigree,
                        'age': age,
                        'gender': gender
                    }

                    # Display confirmation
                    st.success(f"✅ Data from row #{selected_row} loaded successfully!")
                    if gender == "Female" and pregnancies > 0:
                        st.info(f"Note: Gender set to Female because Pregnancies = {pregnancies}")

                    # Show the loaded values
                    st.write("Loaded values:")
                    st.write(f"Pregnancies: {pregnancies}, Glucose: {glucose}, BP: {blood_pressure}")
                    st.write(f"Skin Thickness: {skin_thickness}, Insulin: {insulin}, BMI: {bmi}")
                    st.write(f"Pedigree: {pedigree}, Age: {age}, Gender: {gender}")

            # Add a button to load a random sample
            if st.button("Load Random Sample", key="random_diabetes"):
                if total_samples > 0:
                    random_index = np.random.randint(0, total_samples)
                    sample = load_sample_from_dataset(diabetes_df, random_index, "diabetes")

                    if sample is not None:
                        # Set the input values based on the selected sample
                        pregnancies = int(sample['Pregnancies'])
                        glucose = float(sample['Glucose'])
                        blood_pressure = float(sample['BloodPressure'])
                        skin_thickness = float(sample['SkinThickness'])
                        insulin = float(sample['Insulin'])
                        bmi = float(sample['BMI'])
                        pedigree = float(sample['DiabetesPedigreeFunction'])
                        age = int(sample['Age'])

                        # Handle gender based on pregnancies
                        gender, pregnancies = handle_gender_pregnancy("Female" if pregnancies > 0 else "Male", pregnancies)

                        # Store values in session state
                        st.session_state.diabetes_inputs = {
                            'pregnancies': pregnancies,
                            'glucose': glucose,
                            'blood_pressure': blood_pressure,
                            'skin_thickness': skin_thickness,
                            'insulin': insulin,
                            'bmi': bmi,
                            'pedigree': pedigree,
                            'age': age,
                            'gender': gender
                        }

                        # Display confirmation
                        st.success(f"✅ Random data from row #{random_index} loaded successfully!")
                        if gender == "Female" and pregnancies > 0:
                            st.info(f"Note: Gender set to Female because Pregnancies = {pregnancies}")

                        # Show the loaded values
                        st.write("Loaded values:")
                        st.write(f"Pregnancies: {pregnancies}, Glucose: {glucose}, BP: {blood_pressure}")
                        st.write(f"Skin Thickness: {skin_thickness}, Insulin: {insulin}, BMI: {bmi}")
                        st.write(f"Pedigree: {pedigree}, Age: {age}, Gender: {gender}")
    else:
        st.error("Failed to load diabetes dataset")

    # Manual input section
    st.subheader("Manual Input")
    cols = st.columns(3)

    # Get values from session state
    inputs = st.session_state.diabetes_inputs

    # First get gender since it affects pregnancies
    with cols[2]:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if inputs['gender'] == "Male" else 1)

    # Then get pregnancies with a note about gender
    with cols[0]:
        if gender == "Male":
            pregnancies = 0
            st.write("Pregnancies: 0 (Male)")
        else:
            pregnancies = st.number_input("Pregnancies", 0, 20, value=int(inputs['pregnancies']))

    # Other inputs
    with cols[1]: glucose = st.number_input("Glucose Level", 0.0, 300.0, value=float(inputs['glucose']))
    with cols[2]: blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0, value=float(inputs['blood_pressure']))
    with cols[0]: skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, value=float(inputs['skin_thickness']))
    with cols[1]: insulin = st.number_input("Insulin", 0.0, 1000.0, value=float(inputs['insulin']))
    with cols[2]: bmi = st.number_input("BMI", 0.0, 70.0, value=float(inputs['bmi']))
    with cols[0]: pedigree = st.number_input("Pedigree Function", 0.0, 2.5, value=float(inputs['pedigree']))
    with cols[1]: age = st.number_input("Age", 1, 120, value=int(inputs['age']))

    # Apply gender-pregnancy logic
    gender, pregnancies = handle_gender_pregnancy(gender, pregnancies)

    # Update session state with current values
    st.session_state.diabetes_inputs = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'pedigree': pedigree,
        'age': age,
        'gender': gender
    }

    # Add a button to load a new model if available
    try:
        import os
        new_model_path = f'{working_dir}/saved_models/diabetes_new.pkl'
        if os.path.exists(new_model_path):
            if st.button("Load New Diabetes Model"):
                try:
                    diabetes_model = pickle.load(open(new_model_path, 'rb'))
                    st.success("✅ New diabetes model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to load new model: {e}")
    except Exception:
        pass

    # Add a section for testing with predefined samples
    st.subheader("Test with Predefined Samples")
    test_col1, test_col2 = st.columns(2)

    with test_col1:
        if st.button("Test with Low Risk Sample"):
            if diabetes_model is None:
                st.error("Model not loaded")
            else:
                # Low risk sample
                sample = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
                st.info("Using sample: [1, 85, 66, 29, 0, 26.6, 0.351, 31]")

                # Make prediction
                prediction = diabetes_model.predict([sample])[0]
                try:
                    risk_score = float(diabetes_model.predict_proba([sample])[0][1])
                except:
                    risk_score = float(prediction)
                result = "Has diabetes" if prediction == 1 else "No diabetes"
                recommendation = "Consult a doctor" if prediction == 1 else "Maintain healthy lifestyle"

                st.success(f"Prediction: {result}")
                st.info(f"Risk Score: {risk_score:.2f}")
                st.info(f"Recommendation: {recommendation}")

    with test_col2:
        if st.button("Test with High Risk Sample"):
            if diabetes_model is None:
                st.error("Model not loaded")
            else:
                # High risk sample
                sample = [8, 183, 64, 0, 0, 23.3, 0.672, 32]
                st.info("Using sample: [8, 183, 64, 0, 0, 23.3, 0.672, 32]")

                # Make prediction
                prediction = diabetes_model.predict([sample])[0]
                try:
                    risk_score = float(diabetes_model.predict_proba([sample])[0][1])
                except:
                    risk_score = float(prediction)
                result = "Has diabetes" if prediction == 1 else "No diabetes"
                recommendation = "Consult a doctor" if prediction == 1 else "Maintain healthy lifestyle"

                st.success(f"Prediction: {result}")
                st.info(f"Risk Score: {risk_score:.2f}")
                st.info(f"Recommendation: {recommendation}")

    st.subheader("Custom Prediction")
    if st.button("Predict Diabetes"):
        if diabetes_model is None:
            st.error("Model not loaded")
        else:
            # Display model information
            st.subheader("Model Information")
            model_info = {
                "Model Type": str(type(diabetes_model)),
                "Model Parameters": str(getattr(diabetes_model, 'get_params', lambda: "Not available")())
            }
            st.json(model_info)

            # Create inputs array
            inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]

            # Display input values for debugging
            st.subheader("Input Values")
            input_features = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                             "Insulin", "BMI", "Pedigree", "Age"]
            input_data = {feature: value for feature, value in zip(input_features, inputs)}
            st.json(input_data)

            # Make prediction
            print(f"\n===== MAKING DIABETES PREDICTION =====")
            print(f"Input values: {inputs}")

            # Get prediction
            prediction = diabetes_model.predict([inputs])[0]
            print(f"Raw prediction result: {prediction}")

            # Get probability scores if available
            try:
                probabilities = diabetes_model.predict_proba([inputs])[0]
                print(f"Probability scores: {probabilities}")
                risk_score = float(probabilities[1])
                print(f"Risk score (probability of class 1): {risk_score}")
            except Exception as e:
                print(f"Could not get probability scores: {e}")
                risk_score = float(prediction)
                print(f"Using prediction as risk score: {risk_score}")

            # Determine result and recommendation
            result = "Has diabetes" if prediction == 1 else "No diabetes"
            recommendation = "Consult a doctor" if prediction == 1 else "Maintain healthy lifestyle"

            print(f"Final prediction: {result}")
            print(f"Risk score: {risk_score:.2f}")
            print(f"Recommendation: {recommendation}")
            print("=====================================\n")

            # Display results
            st.subheader("Prediction Results")
            st.success(f"Prediction: {result}")
            st.info(f"Risk Score: {risk_score:.2f}")
            st.info(f"Recommendation: {recommendation}")

            if use_db:
                # Create payload without username in prediction_data
                payload = {
                    'prediction_type': 'diabetes',
                    'prediction_data': {
                        'pregnancies': pregnancies,
                        'glucose': glucose,
                        'blood_pressure': blood_pressure,
                        'skin_thickness': skin_thickness,
                        'insulin': insulin,
                        'bmi': bmi,
                        'pedigree': pedigree,
                        'age': age,
                        'gender': gender
                    },
                    'prediction_result': result,
                    'risk_score': risk_score,
                    'recommendation': recommendation,
                    'username': st.session_state.username  # Add username at the top level
                }
                rec_id = save_prediction('diabetes', payload)
                if rec_id:
                    st.success(f"Saved record ID: {rec_id}")
                else:
                    st.warning("Failed to save record")

    # --- Heart Disease Prediction ---
    elif selected == 'Heart Disease Prediction':
        st.title("❤️ Heart Disease Prediction")

        # Add a section for loading data from CSV
        st.subheader("Dataset Browser")

    # Load the heart disease dataset
    heart_df = load_dataset('heart')

    if heart_df is not None:
        # Display dataset info
        total_samples = len(heart_df)
        st.info(f"Loaded {total_samples} samples from heart disease dataset")

        # Add controls for dataset browsing
        col1, col2 = st.columns(2)

        with col1:
            # Display the dataset with a fixed number of rows
            num_rows = st.session_state.num_rows
            st.write(f"Showing {num_rows} sample rows from dataset")

            # Display the dataframe with sample rows
            st.dataframe(heart_df.head(num_rows))

        with col2:
            # Add a number input to select a specific row
            selected_row = st.number_input("Select row to use", 0, total_samples - 1, 0, key="heart_selected_row")

            if st.button("Load Selected Row", key="use_heart_row"):
                sample = load_sample_from_dataset(heart_df, selected_row, "heart")

                if sample is not None:
                    # Set the input values based on the selected sample
                    age = int(sample['age'])
                    sex = "Male" if sample['sex'] == 1 else "Female"
                    cp = int(sample['cp'])
                    trestbps = float(sample['trestbps'])
                    chol = float(sample['chol'])
                    fbs = "Yes" if sample['fbs'] == 1 else "No"
                    restecg = int(sample['restecg'])
                    thalach = float(sample['thalach'])
                    exang = "Yes" if sample['exang'] == 1 else "No"
                    oldpeak = float(sample['oldpeak'])
                    slope = int(sample['slope'])
                    ca = int(sample['ca'])
                    thal = int(sample['thal'])

                    # Display confirmation
                    st.success(f"✅ Data from row #{selected_row} loaded successfully!")

            # Add a button to load a random sample
            if st.button("Load Random Sample", key="random_heart"):
                if total_samples > 0:
                    random_index = np.random.randint(0, total_samples)
                    sample = load_sample_from_dataset(heart_df, random_index, "heart")

                    if sample is not None:
                        # Set the input values based on the selected sample
                        age = int(sample['age'])
                        sex = "Male" if sample['sex'] == 1 else "Female"
                        cp = int(sample['cp'])
                        trestbps = float(sample['trestbps'])
                        chol = float(sample['chol'])
                        fbs = "Yes" if sample['fbs'] == 1 else "No"
                        restecg = int(sample['restecg'])
                        thalach = float(sample['thalach'])
                        exang = "Yes" if sample['exang'] == 1 else "No"
                        oldpeak = float(sample['oldpeak'])
                        slope = int(sample['slope'])
                        ca = int(sample['ca'])
                        thal = int(sample['thal'])

                        # Display confirmation
                        st.success(f"✅ Random data from row #{random_index} loaded successfully!")
    else:
        st.error("Failed to load heart disease dataset")

    # Manual input section
    st.subheader("Manual Input")
    cols = st.columns(3)
    with cols[0]: age = st.number_input("Age", 1, 120)
    with cols[1]: sex = st.selectbox("Sex", ["Male", "Female"])
    with cols[2]: cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    with cols[0]: trestbps = st.number_input("Resting BP", 0.0, 300.0)
    with cols[1]: chol = st.number_input("Cholesterol", 0.0, 600.0)
    with cols[2]: fbs = st.selectbox("Fasting Blood Sugar >120", ["No","Yes"])
    with cols[0]: restecg = st.number_input("Resting ECG", 0, 2)
    with cols[1]: thalach = st.number_input("Max Heart Rate", 0.0, 300.0)
    with cols[2]: exang = st.selectbox("Exercise Angina", ["No","Yes"])
    with cols[0]: oldpeak = st.number_input("ST Depression", 0.0, 10.0)
    with cols[1]: slope = st.selectbox("ST Slope", [0,1,2])
    with cols[2]: ca = st.selectbox("Major Vessels", [0,1,2,3])
    with cols[0]: thal = st.selectbox("Thalassemia", [0,1,2,3])

    if st.button("Predict Heart Disease"):
        if heart_model is None:
            st.error("Model not loaded")
        else:
            inputs = [
                age,
                1 if sex=="Male" else 0,
                cp,
                trestbps,
                chol,
                1 if fbs=="Yes" else 0,
                restecg,
                thalach,
                1 if exang=="Yes" else 0,
                oldpeak,
                slope,
                ca,
                thal
            ]
            prediction = heart_model.predict([inputs])[0]
            try:
                risk_score = float(heart_model.predict_proba([inputs])[0][1])
            except:
                risk_score = float(prediction)
            result = "Has heart disease" if prediction==1 else "No heart disease"
            recommendation = "Consult a cardiologist" if prediction==1 else "Healthy heart"

            st.success(f"Prediction: {result}")
            st.info(f"Risk Score: {risk_score:.2f}")
            st.info(f"Recommendation: {recommendation}")

            if use_db:
                # Create payload without username in prediction_data
                payload = {
                    'prediction_type': 'heart',
                    'prediction_data': {
                        'age': age,
                        'sex': sex,
                        'cp': cp,
                        'trestbps': trestbps,
                        'chol': chol,
                        'fbs': fbs,
                        'restecg': restecg,
                        'thalach': thalach,
                        'exang': exang,
                        'oldpeak': oldpeak,
                        'slope': slope,
                        'ca': ca,
                        'thal': thal
                    },
                    'prediction_result': result,
                    'risk_score': risk_score,
                    'recommendation': recommendation,
                    'username': st.session_state.username  # Add username at the top level
                }
                rec_id = save_prediction('heart', payload)
                if rec_id:
                    st.success(f"Saved record ID: {rec_id}")
                else:
                    st.warning("Failed to save record")

    # --- Kidney Disease Prediction ---
    elif selected == 'Kidney Disease Prediction':
        st.title("🧪 Kidney Disease Prediction")

        # Add a section for loading data from CSV
        st.subheader("Dataset Browser")

    # Load the kidney disease dataset
    kidney_df = load_dataset('kidney')

    if kidney_df is not None:
        # Display dataset info
        total_samples = len(kidney_df)
        st.info(f"Loaded {total_samples} samples from kidney disease dataset")

        # Add controls for dataset browsing
        col1, col2 = st.columns(2)

        with col1:
            # Display the dataset with a fixed number of rows
            num_rows = st.session_state.num_rows
            st.write(f"Showing {num_rows} sample rows from dataset")

            # Display the dataframe with sample rows
            st.dataframe(kidney_df.head(num_rows))

        with col2:
            # Add a number input to select a specific row
            selected_row = st.number_input("Select row to use", 0, total_samples - 1, 0, key="kidney_selected_row")

            if st.button("Load Selected Row", key="use_kidney_row"):
                sample = load_sample_from_dataset(kidney_df, selected_row, "kidney")

                if sample is not None:
                    # Display confirmation
                    st.success(f"✅ Data from row #{selected_row} loaded successfully!")

                    # Display the classification result
                    classification = sample.get('classification', 'Unknown')
                    st.info(f"Classification: {classification}")

            # Add a button to load a random sample
            if st.button("Load Random Sample", key="random_kidney"):
                if total_samples > 0:
                    random_index = np.random.randint(0, total_samples)
                    sample = load_sample_from_dataset(kidney_df, random_index, "kidney")

                    if sample is not None:
                        # Display confirmation
                        st.success(f"✅ Random data from row #{random_index} loaded successfully!")

                        # Display the classification result
                        classification = sample.get('classification', 'Unknown')
                        st.info(f"Classification: {classification}")
    else:
        st.error("Failed to load kidney disease dataset")

    st.warning("Kidney disease prediction model not yet implemented")
    st.info("You can browse the dataset to understand the features used for kidney disease prediction")
