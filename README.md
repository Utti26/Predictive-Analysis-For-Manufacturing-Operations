# Predictive Analysis for Manufacturing Operations

This project provides a RESTful API to predict machine downtime using manufacturing data. The API includes endpoints to upload a dataset, train a machine learning model, and make predictions.

---

## **Features**
- **Upload Endpoint**: Upload a CSV file containing manufacturing data.
- **Train Endpoint**: Train a machine learning model (Decision Tree Classifier) on the uploaded data.
- **Predict Endpoint**: Provide inputs like `Machine_ID`, `Temperature`, and `Run_Time` to predict downtime.

---

## **Setup Instructions**

### Prerequisites
- Python 3.7 or higher
- Required Libraries: `Flask`, `pandas`, `scikit-learn`, `numpy`

### Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_name>
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Generate synthetic data (optional): If no dataset is available, generate a sample dataset using the script:

bash
Copy
Edit
python create_dataset.py
Run the API:

bash
Copy
Edit
python app.py
API Endpoints
1. Upload Data
Endpoint: POST /upload
Description: Upload a CSV file with columns Machine_ID, Temperature, Run_Time, and Downtime_Flag.
Input: CSV file
Response:
json
Copy
Edit
{
  "message": "File uploaded and processed successfully."
}
Error:
json
Copy
Edit
{
  "error": "No file provided."
}
2. Train Model
Endpoint: POST /train
Description: Train the model using the uploaded dataset.
Response:
json
Copy
Edit
{
  "accuracy": 0.95,
  "classification_report": {
    "0": {
      "precision": 0.96,
      "recall": 0.98,
      "f1-score": 0.97,
      "support": 20
    },
    "1": {
      "precision": 0.89,
      "recall": 0.83,
      "f1-score": 0.86,
      "support": 10
    }
  }
}
3. Predict Downtime
Endpoint: POST /predict
Description: Provide Machine_ID, Temperature, and Run_Time to predict downtime.
Input:
json
Copy
Edit
{
  "Machine_ID": "M1",
  "Temperature": 85.5,
  "Run_Time": 5.2
}
Response:
json
Copy
Edit
{
  "Downtime": "Yes",
  "Confidence": 0.88
}
File Structure
bash
Copy
Edit
.
├── app.py                  # Main API code
├── create_dataset.py       # Script to generate synthetic data
├── model.py                # Model training and evaluation script
├── manufacturing_data.csv  # Example dataset (auto-generated)
└── README.md               # Documentation
Testing the API
Use Postman or cURL to test the API locally:

Upload Data:

bash
Copy
Edit
curl -X POST -F "file=@manufacturing_data.csv" http://127.0.0.1:5000/upload
Train Model:

bash
Copy
Edit
curl -X POST http://127.0.0.1:5000/train
Make Predictions:

bash
Copy
Edit
curl -X POST -H "Content-Type: application/json" -d '{"Machine_ID": "M1", "Temperature": 85, "Run_Time": 5}' http://127.0.0.1:5000/predict
Future Enhancements
Add advanced preprocessing and feature selection.
Support for other machine learning models.
Implement authentication and user access control for the API.
