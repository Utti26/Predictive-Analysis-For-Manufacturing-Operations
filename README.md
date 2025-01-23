# Predictive-Analysis-For-Manufacturing-Operations

This is a Flask web application that enables users to:
- Upload a dataset.
- Train a machine learning model (Decision Tree Classifier) on the uploaded dataset.
- Use the trained model to make predictions based on user inputs.

## Features
1. **File Upload (`/upload`)**:
   - Accepts a CSV file containing machinery data.
   - Encodes the `Machine_ID` column into numerical values using Label Encoding.

2. **Model Training (`/train`)**:
   - Trains a Decision Tree Classifier on the uploaded dataset.
   - Splits the data into training and testing sets (80%-20%).
   - Evaluates the model and returns:
     - **Accuracy Score**: The proportion of correct predictions.
     - **Classification Report**: Precision, recall, and F1-score for each class.

3. **Prediction (`/predict`)**:
   - Accepts inputs: `Temperature`, `Run_Time`, and `Machine_ID`.
   - Encodes `Machine_ID` and predicts whether downtime will occur (Yes/No).
   - Returns the prediction along with confidence probability.

---

## How to Run the Application

### Prerequisites
Ensure you have Python installed along with the required libraries. Use the following command to install the dependencies:

```bash
pip install flask pandas scikit-learn
```

### Steps
1. Clone this repository or copy the code.
2. Save the code in a file, e.g., `app.py`.
3. Run the application using:

```bash
python app.py
```

4. Access the application at `http://127.0.0.1:5000` in your browser or API testing tools like Postman.

---

## API Endpoints

### 1. **Upload Dataset**
**Endpoint**: `/upload`  
**Method**: `POST`

#### Request:
- Upload a CSV file containing the dataset. Example dataset columns:
  - `Machine_ID`: Categorical identifier for the machine.
  - `Temperature`: Numerical value indicating machine temperature.
  - `Run_Time`: Numerical value indicating runtime.
  - `Downtime_Flag`: Target variable (1 = Downtime, 0 = No Downtime).

#### Response:
- Success: `{"message": "File uploaded and processed successfully."}`
- Error: `{"error": "<error message>"}`

### 2. **Train Model**
**Endpoint**: `/train`  
**Method**: `POST`

#### Request:
No additional data required. The model uses the uploaded dataset.

#### Response:
- Success:
  ```json
  {
    "accuracy": 0.95,
    "classification_report": {
      "0": {"precision": 0.96, "recall": 0.94, "f1-score": 0.95},
      "1": {"precision": 0.92, "recall": 0.97, "f1-score": 0.94},
      "accuracy": 0.95
    }
  }
  ```
- Error: `{"error": "<error message>"}`

### 3. **Predict Downtime**
**Endpoint**: `/predict`  
**Method**: `POST`

#### Request:
Provide the following JSON data:
```json
{
  "Machine_ID": "M1",
  "Temperature": 75.3,
  "Run_Time": 120
}
```

#### Response:
- Success:
  ```json
  {
    "Downtime": "Yes",
    "Confidence": 0.85
  }
  ```
- Error: `{"error": "<error message>"}`

---

## Example Dataset
| Machine_ID | Temperature | Run_Time | Downtime_Flag |
|------------|-------------|----------|---------------|
| M1         | 75.3        | 120      | 1             |
| M2         | 65.0        | 80       | 0             |
| M1         | 70.0        | 100      | 1             |

---

## Code Overview
1. **Flask Initialization**:
   - The Flask app is initialized with `Flask(__name__)`.

2. **Global Variables**:
   - `uploaded_data`: Stores the uploaded dataset.
   - `model`: Stores the trained Decision Tree Classifier.
   - `label_encoder`: Stores the encoder for `Machine_ID`.

3. **Endpoints**:
   - `/upload`: Handles file upload and preprocessing.
   - `/train`: Trains the model using the uploaded data.
   - `/predict`: Predicts downtime based on user inputs.

4. **Error Handling**:
   - Each endpoint has a `try-except` block to handle exceptions and return appropriate error messages.

---

## Dependencies
- Flask
- Pandas
- Scikit-learn

---

## Future Enhancements
- Add support for other machine learning algorithms.
- Enable data visualization for exploratory data analysis (EDA).
- Deploy the app on a cloud platform like AWS, Azure, or Google Cloud.

---

## License
This project is free to use for educational purposes. Feel free to modify and enhance it as needed.

