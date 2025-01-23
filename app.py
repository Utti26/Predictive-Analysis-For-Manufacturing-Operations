#utkarshPalTechPraneeAssignment
# Import necessary libraries
from flask import Flask, request, jsonify  # Flask for creating API endpoints
import pandas as pd  # Pandas for data handling
from sklearn.tree import DecisionTreeClassifier  # Decision Tree for training a model
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.metrics import accuracy_score, classification_report  # For evaluating the model
from sklearn.preprocessing import LabelEncoder  # To encode categorical variables
import os  # To interact with the operating system (not used in the code but imported)

# Initialize Flask app
app = Flask(__name__)

# Global variables to store uploaded data, trained model, and label encoder
uploaded_data = None
model = None
label_encoder = None

# Route for uploading and processing data
@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_data, label_encoder  # Use global variables to store the data and encoder

    # Check if a file is provided in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    # Access the uploaded file
    file = request.files['file']

    try:
        # Load the file into a pandas DataFrame
        uploaded_data = pd.read_csv(file)

        # Encode the 'Machine_ID' column into numeric values using LabelEncoder
        label_encoder = LabelEncoder()
        uploaded_data['Machine_ID_Encoded'] = label_encoder.fit_transform(uploaded_data['Machine_ID'])

        return jsonify({"message": "File uploaded and processed successfully."}), 200
    except Exception as e:
        # Return an error if something goes wrong
        return jsonify({"error": str(e)}), 500

# Route for training the model
@app.route('/train', methods=['POST'])
def train():
    global uploaded_data, model  # Use global variables for data and model

    # Check if data has been uploaded
    if uploaded_data is None:
        return jsonify({"error": "No data uploaded. Please upload a dataset first."}), 400

    try:
        # Define features (X) and target (y) for training
        X = uploaded_data[['Machine_ID_Encoded', 'Temperature', 'Run_Time']]
        y = uploaded_data['Downtime_Flag']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train a Decision Tree Classifier
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Test the model and calculate predictions
        y_pred = model.predict(X_test)

        # Evaluate the model using accuracy score and classification report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Return evaluation results as a JSON response
        return jsonify({
            "accuracy": accuracy,
            "classification_report": report
        }), 200
    except Exception as e:
        # Return an error if something goes wrong
        return jsonify({"error": str(e)}), 500

# Route for making predictions using the trained model
@app.route('/predict', methods=['POST'])
def predict():
    global model, label_encoder  # Use global variables for the model and encoder

    # Check if the model has been trained
    if model is None:
        return jsonify({"error": "No model trained. Please train the model first."}), 400

    try:
        # Get input data from the request in JSON format
        input_data = request.get_json()
        temperature = input_data.get("Temperature")
        run_time = input_data.get("Run_Time")
        machine_id = input_data.get("Machine_ID")

        # Ensure all required fields are provided
        if temperature is None or run_time is None or machine_id is None:
            return jsonify({"error": "Missing required fields (Temperature, Run_Time, Machine_ID)."}), 400

        # Encode the Machine_ID using the previously created label encoder
        machine_id_encoded = label_encoder.transform([machine_id])[0]

        # Prepare input features for the model
        input_features = [[machine_id_encoded, temperature, run_time]]

        # Make predictions using the trained model
        prediction = model.predict(input_features)  # Predict whether there will be downtime
        probability = model.predict_proba(input_features)[0][1]  # Get the probability of the predicted class

        # Interpret the prediction (1 = Yes, 0 = No)
        downtime = "Yes" if prediction[0] == 1 else "No"

        # Return the prediction and confidence as a JSON response
        return jsonify({"Downtime": downtime, "Confidence": round(probability, 2)}), 200
    except Exception as e:
        # Return an error if something goes wrong
        return jsonify({"error": str(e)}), 500

# Main entry point to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
