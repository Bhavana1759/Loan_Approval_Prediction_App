# 💳 Loan Approval Prediction App
A **Streamlit web app** that predicts whether a loan application will be approved or not based on applicant details.  
This project uses **Machine Learning (Random Forest)** trained on applicant data.

---

## 🚀 Features
- Upload or load loan dataset for training.  
- Preprocesses categorical and numerical data automatically.  
- Trains a **Random Forest Classifier** for loan approval prediction.  
- Interactive UI for entering applicant details.  
- Displays prediction result along with probability score.  

---
## 📂 Project Structure
Loan_Approval_Prediction_App/
│── app.py # Main Streamlit application

│── credtech_data.csv # Dataset (must include loan_status column)

│── requirements.txt # Python dependencies

│── README.md # Project documentation

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Loan_Approval_Prediction_App.git
   cd Loan_Approval_Prediction_App

2. (Optional) Create a virtual environment:
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
3.Install dependencies:
   pip install -r requirements.txt
▶️ Run the App:
   streamlit run app.py
---

The app will start on http://localhost:8501

---
📊 Dataset

The dataset file should be named credtech_data.csv.

It must contain a target column named loan_status (e.g., Approved / Rejected).

Example columns:

age

income

credit_score

loan_amount

loan_term

employment_years

loan_status

🛠️ Technologies Used

Python

Streamlit

Pandas & NumPy

Scikit-learn (RandomForestClassifier)

📜 License

This project is licensed under the MIT License.

🙌 Acknowledgments

Streamlit---->for interactive UI

Scikit-learn---->for ML model

Would you like me to also create a **`requirements.txt`** file for you (so you can run `pip install -r requirements.txt` easily)?
