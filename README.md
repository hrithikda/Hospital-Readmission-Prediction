README.md
markdown
Copy
Edit
# 🏥 Hospital Readmission Prediction

**Hospital Readmission Prediction** is a data science project that uses machine learning techniques to predict **Excess Readmission Ratios** for hospitals. The goal is to assist healthcare providers in identifying key factors leading to patient readmissions and optimizing strategies to improve patient care and hospital performance.

---

##  **Project Overview**

Hospital readmissions are a critical metric for evaluating healthcare quality. Readmission penalties, like those under the **CMS Hospital Readmissions Reduction Program (HRRP)**, have made it essential for hospitals to track and minimize avoidable readmissions. 

This project involves:
- Cleaning and analyzing a public dataset of hospital metrics.
- Building machine learning models to predict the **Excess Readmission Ratio**, which compares a hospital's readmission performance to the national average.
- Using advanced techniques like **SHAP (SHapley Additive Explanations)** to explain the contributions of individual features to model predictions.

---

##  **Problem Statement**

### **Objective:**
Predict whether hospitals have **excess readmission rates** and understand the **key drivers** of these predictions.

### **Challenges:**
1. Handling missing data across several key columns.
2. Encoding high-cardinality categorical features like hospital names.
3. Balancing interpretability and prediction accuracy by evaluating multiple models.

---

##  **Dataset Description**

The project is based on the **FY 2024 CMS Hospital Readmissions Reduction Program dataset**. It includes metrics for ~18,000 hospitals across the US.

### **Key Features:**
- **Facility Name**: Name of the hospital.
- **State**: The state where the hospital is located.
- **Number of Discharges**: The number of patient discharges (proxy for hospital size).
- **Predicted Readmission Rate**: The expected rate of readmissions based on hospital characteristics.
- **Expected Readmission Rate**: Benchmark rate for comparison.
- **Excess Readmission Ratio**: Target variable, comparing hospital performance to the national average.

### **Target Variable:**
- **Excess Readmission Ratio**: A value >1 indicates higher-than-expected readmissions, while <1 indicates better-than-expected performance.

---

##  **Solution Workflow**

### **1️⃣ Data Cleaning**
- **Missing Values**:
  - Numerical columns were imputed using **KNN Imputation** for better accuracy.
  - Categorical columns like "Number of Readmissions" were converted to numeric, treating `"Too Few to Report"` as missing.
- **Date Features**: Converted `Start Date` and `End Date` into year-based numerical features.

### **2️⃣ Feature Engineering**
- **Encoding Categorical Variables**:
  - High-cardinality variables like `Facility Name` were encoded using **Target Encoding**.
  - Categorical features like `State` were **Label Encoded**.
  - Features with a small number of categories, such as `Measure Name`, were **One-Hot Encoded**.

### **3️⃣ Model Training**
Five models were trained for comparative analysis:
1. **Linear Regression**: Baseline model for benchmarking.
2. **Ridge Regression**: Adds L2 regularization to handle multicollinearity.
3. **Lasso Regression**: Adds L1 regularization to perform feature selection.
4. **XGBoost Regressor**: Advanced tree-based ensemble model for accurate predictions.
5. **Neural Network**: A deep learning model for capturing complex patterns.

### **4️⃣ Explainability (SHAP Analysis)**
- Used **SHAP (SHapley Additive Explanations)** to analyze feature importance and understand how each feature contributes to the model's predictions.

---

##  **Results & Insights**

### **Model Performance**
| Model             | Mean Squared Error (MSE) | R² Score |
|-------------------|--------------------------|----------|
| Linear Regression | 0.00067                  | 0.82051  |
| Ridge Regression  | 0.00067                  | 0.82060  |
| Lasso Regression  | 0.00372                  | -0.00051 |
| XGBoost           | 0.00018                  | 0.95219  |
| Neural Network    | 0.00009                  | 0.97474  |

### **Key Features Identified (via SHAP)**
1. **Predicted Readmission Rate**: The most critical predictor of excess readmission ratios.
2. **Expected Readmission Rate**: A benchmark for evaluating hospital performance.
3. **Number of Discharges**: Larger hospitals tend to have higher readmissions.

---

## 📂 **Project Structure**

📂 Hospital-Readmission-Prediction ├── hospital_readmission_prediction.ipynb # Jupyter Notebook (Main file) ├── 📁 data/ │ └── cleaned_hospital_readmissions.csv # Cleaned dataset ├── 📁 models/ │ ├── xgboost_model.pkl # Saved XGBoost model │ ├── neural_network_model.h5 # Saved Neural Network model ├── train_model.py # Python script for training ├── requirements.txt # Dependencies ├── README.md # Project documentation

yaml
Copy
Edit

---

##  **How to Run the Project**

### **Clone the Repository**
```bash
git clone https://github.com/hrithikda/Hospital-Readmission-Prediction.git
cd Hospital-Readmission-Prediction
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Run the Training Script
bash
Copy
Edit
python train_model.py
Run the Notebook
Open hospital_readmission_prediction.ipynb in Jupyter Notebook or Google Colab.
Follow the steps for data preprocessing, model training, and evaluation.
🛠️ Technologies Used
Languages & Libraries:
Python, Pandas, NumPy, Scikit-learn, TensorFlow, XGBoost, SHAP
Tools:
Google Colab for notebook development
GitHub for version control and sharing
Models:
Regression (Linear, Ridge, Lasso)
Advanced Ensemble Models (XGBoost)
Deep Learning (Neural Networks)
📬 Contact
Author: Hrithik Dasharatha Angadi
GitHub: github.com/hrithikda
LinkedIn: linkedin.com/in/hrithikda
📜 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---
