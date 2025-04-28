# Disease Prediction System

A web application for predicting various diseases using machine learning models.

## Features

- User authentication system
- Diabetes prediction
- Heart disease prediction
- Kidney disease prediction
- Dataset browser for each disease type
- Prediction history storage in database

## Technologies Used

- **Backend**: Django, Django REST Framework
- **Frontend**: Streamlit
- **Database**: SQLite
- **Machine Learning**: Scikit-learn

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd djbackendps
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run database migrations:
   ```
   python manage.py migrate
   ```

### Running the Application

1. Start the Django backend:
   ```
   python manage.py runserver
   ```

2. Start the Streamlit frontend:
   ```
   cd streamlit_app
   python -m streamlit run app.py
   ```

3. Access the application:
   - Django admin: http://localhost:8000/admin
   - Streamlit app: http://localhost:8501

## Usage

1. Log in using one of the following demo accounts:
   - Username: admin, Password: admin123
   - Username: doctor, Password: doctor123
   - Username: nurse, Password: nurse123
   - Username: user, Password: password
   - Username: test, Password: test

2. Select a disease type from the sidebar menu

3. Browse the dataset or enter your own values

4. Click the "Predict" button to get a prediction

5. View the prediction result, risk score, and recommendation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Dataset sources
- Machine learning model references
- Any other acknowledgements
