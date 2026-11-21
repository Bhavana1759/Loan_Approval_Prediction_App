# 💳 Loan Approval Prediction App

A **Java-based desktop application** that predicts whether a loan application will be approved or rejected based on applicant details. The project uses a **Random Forest Machine Learning algorithm** implemented in Java and trained using applicant data from a CSV dataset.

## 🚀 Features

- Loads applicant data automatically from a CSV dataset.
- Displays a preview of the first five rows of the dataset.
- Automatically identifies numerical and categorical features.
- Preprocesses categorical data using label encoding.
- Calculates mean values for numerical features.
- Trains a Random Forest classifier using 50 decision trees.
- Uses bootstrap sampling to train individual decision trees.
- Provides an interactive form to enter applicant details.
- Predicts the loan approval status.
- Displays the prediction result along with its probability score.
- Includes input validation for numerical values.

## 📂 Project Structure

```text
Loan_Approval_Prediction_App/
│
├── App.java              # Main Java application
├── credtech_data.csv     # Dataset containing applicant details
└── README.md             # Project documentation
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Loan_Approval_Prediction_App.git
cd Loan_Approval_Prediction_App
```

### 2. Install Java

Make sure Java JDK is installed on your system.

Check the installation:

```bash
java -version
javac -version
```

### 3. Add the Dataset

Place the `credtech_data.csv` file in the project folder.

The dataset must contain a target column named `loan_status`.

### 4. Compile the Application

```bash
javac App.java
```

### 5. Run the Application

```bash
java App
```

The Loan Approval Prediction desktop application will open.

## 📊 Dataset

The application uses a CSV dataset named `credtech_data.csv`.

**Dataset requirements:**

- Must contain applicant information.
- Must include a target column named `loan_status`.
- The target column contains loan approval categories such as Approved and Rejected.
- All columns except `loan_status` are treated as input features.

The application automatically identifies numerical and categorical columns based on the dataset values.

## 🤖 Machine Learning Algorithm

**Random Forest Classifier**

The project implements a Random Forest algorithm using multiple Decision Trees.

Key implementation details:

- Number of Decision Trees: 50
- Maximum Tree Depth: 8
- Bootstrap Sampling: Enabled
- Feature Selection: Random feature selection using the square root of the total number of features.
- Splitting Criterion: Gini Impurity
- Prediction: Majority voting across the decision trees.
- Probability: Calculated using the proportion of trees voting for each class.

## 🛠️ Technologies Used

- **Java** – Application development and machine learning implementation.
- **Java Swing (AWT)** – Graphical User Interface (GUI).
- **Java Collections** – Data handling and preprocessing.
- **CSV** – Dataset storage and loading.
- **Random Forest** – Loan approval classification.
- **Decision Trees** – Individual prediction models.
- **Gini Impurity** – Decision tree splitting criterion.

## 📜 License

This project is licensed under the MIT License.

## 🙌 Acknowledgments

- Java Swing – For developing the interactive desktop interface.
- Machine Learning – For implementing the Random Forest classification approach.
