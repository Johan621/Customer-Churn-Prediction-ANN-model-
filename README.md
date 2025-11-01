# 🧠 Customer Churn Prediction using ANN  

A complete end-to-end deep learning project built with an **Artificial Neural Network (ANN)** to predict customer churn — determining whether a customer is likely to leave a company or stay, based on their banking data.

---

## 🚀 Project Overview  
This project aims to help businesses retain customers by predicting churn using historical data.  
It uses the `Churn_Modelling.csv` dataset and includes all essential steps:  

- Data preprocessing (encoding, scaling, splitting)
- Building and training an ANN model using **Keras & TensorFlow**
- Evaluating performance with metrics like accuracy and confusion matrix
- Saving trained models and preprocessing objects (`.h5`, `.pkl`)
- Simple prediction script (`app.py`) for real-time inference

This project can serve as a great portfolio example demonstrating your skills in **Machine Learning, Deep Learning, and Model Deployment**.

---

## 📁 Repository Structure  
Customer-Churn-Prediction-ANN-model-/
│
├── Churn_Modelling.csv # Dataset
├── experiments.ipynb # Data preprocessing + model building notebook
├── prediction.ipynb # Example predictions using saved model
├── app.py # Script to run predictions
├── model.h5 # Trained ANN model
├── label_encoder_gender.pkl # Label encoder for Gender feature
├── onehot_encoder_geography.pkl # One-hot encoder for Geography
├── scaler.pkl # Feature scaler (StandardScaler or MinMaxScaler)
├── requirements.txt # List of dependencies
└── README.md # Project documentation

---

## ✅ Features  
- End-to-end ANN model to predict customer churn  
- Preprocessing using LabelEncoder, OneHotEncoder, and Scaler  
- Model serialization (`.h5` and `.pkl` files for reuse)  
- Confusion matrix and accuracy evaluation  
- Real-time prediction with sample customer input  
- Easy retraining using Jupyter notebook  

---

## 🔧 Setup & Installation  

1. **Clone the repository**
   ```bash
   git clone https://github.com/Johan621/Customer-Churn-Prediction-ANN-model-.git
   cd Customer-Churn-Prediction-ANN-model-
2. **Create and activate a virtual environment (optional but recommended)**
```bash
python -m venv venv
venv\Scripts\activate   # For Windows
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Ensure the dataset Churn_Modelling.csv is present in the root directory.**
5. **To retrain the model, open experiments.ipynb and run all cells.**
6. **To make predictions, you can use:**
   ```bash
   python app.py
   ```
   or open prediction.ipynb to test manually.

📊 Model Architecture

Input Layer: Number of neurons = number of features after encoding

Hidden Layers: Dense layers with ReLU activation

Output Layer: Single neuron with Sigmoid activation for binary classification

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

| Metric             | Value (Example)        |
| ------------------ | ---------------------- |
| Accuracy           | 85% – 88%              |
| Loss               | ~0.32                  |
| Precision / Recall | Balanced               |
| Confusion Matrix   | Visualized in notebook |

**Insights:**

Customers with low tenure, low credit score, or fewer products are more likely to churn.

Long-tenure, multi-product customers with higher balance tend to stay.
**🎯 How to Use the Model**

1. **Load the saved model:**
```bash
from tensorflow.keras.models import load_model
import joblib
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

```
2. **Prepare new customer data with the same features as training data.**

3. **Apply the same encoders and scaler.**

4. **Predict churn:**
```bash

prediction = model.predict(new_data)
churn_status = 'Yes' if prediction > 0.5 else 'No'
print(churn_status)
```
**🙋‍♂️ Author & Credits**

**Author:** Johan621

**Dataset:** Banking Customer Churn Dataset (Churn_Modelling.csv)
**Frameworks:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy


**📄 License**

This project is licensed under the MIT License — see the LICENSE
 file for details.
