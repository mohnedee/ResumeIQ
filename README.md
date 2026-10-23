ResumeIQ

NLP-Based Resume (CV) Classification and Web-Based Human Resources
Management System

1. Project Overview

ResumeIQ is a web-based application that combines Natural Language
Processing (NLP), machine learning, and human resources management. It is
designed to classify resumes into job-related categories and provide
separate interfaces for candidates and HR personnel.

The system aims to support the initial resume-screening process by
automatically predicting a resume’s category and providing tools to
manage and review submitted resumes.

2. Key Features

* AI-Based Resume Classification: Predicts a job-related category from
resume text using a trained machine learning model.
* PDF Resume Processing: Accepts PDF resumes and extracts text for
classification.
* Candidate Accounts: Supports candidate registration, login, resume
submission, and access to personal resume history.
* HR Dashboard: Allows authorized HR personnel to review submitted
resumes and use search and category filters.
* Role-Based Access Control: Restricts candidate and HR features
according to user roles.
* Confidence Score: Displays a model confidence score when available.
* Dashboard Analytics: Provides visual summaries of resume categories and
upload activity.
* Relational Database: Stores application data, including user accounts
and resume-related records.
* Automated Tests: Includes tests for API endpoints, authentication, and
access permissions.

3. Technologies Used

Component	Technologies
Programming Language	Python
Web Framework	FastAPI
Machine Learning	Scikit-learn
Natural Language Processing	TF-IDF text vectorization
Classification	Logistic Regression, Linear SVM, Multinomial Naive Bayes
Data Processing	Pandas, NumPy
Model Persistence	Joblib
Database	SQLite with SQLAlchemy
Frontend	HTML, CSS, JavaScript
PDF Processing	pypdf
Testing	Pytest, FastAPI TestClient

4. Machine Learning Workflow

The machine learning workflow includes:

1. Loading and examining the resume dataset.
2. Cleaning the dataset and removing duplicate resume texts.
3. Splitting the data into training and testing sets using stratified
sampling.
4. Converting resume text into numerical features using TF-IDF.
5. Training and comparing multiple classification algorithms.
6. Evaluating model performance using classification metrics and a
confusion matrix.
7. Saving the selected model for use by the web application.

The model predicts a resume category based on patterns learned from the
training data. Its output is intended to assist the screening process and
should not replace human review.

5. Application Architecture

The application connects a web interface to a FastAPI backend. The
backend handles authentication, role-based authorization, resume
processing, model inference, and database operations.

The candidate interface supports resume submission and personal history,
while the HR interface provides access to submitted resumes and dashboard
analytics, subject to role permissions.

6. Project Structure

WEX428/
├── data/                 # Resume dataset
├── models/               # Trained machine learning models
├── reports/              # Evaluation reports and metrics
├── src/                  # API, authentication, and application logic
├── static/                # CSS and frontend assets
├── templates/             # HTML templates
├── uploads/               # Local resume uploads (not committed)
├── create_hr.py           # HR account creation utility
├── requirements.txt       # Python dependencies
├── test_api.py            # API tests
├── test_auth.py           # Authentication tests
├── test_auth_service.py   # Authentication service tests
├── test_extract.py        # PDF extraction test script
└── README.md

7. Installation and Setup

Prerequisites

* Python 3.9 or a compatible Python environment.
* Git.
* A terminal.

Installation

Clone the repository and enter the project directory:

git clone <REPOSITORY_URL>
cd WEX428

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install the dependencies:

pip install -r requirements.txt

Ensure that the required dataset and trained model files are available in
their expected directories before running the application.

Run the Application

From the project root, run:

uvicorn src.api:app --reload

Open the following address in a browser:

* Web application: http://127.0.0.1:8000/
* Interactive API documentation: http://127.0.0.1:8000/docs
* Health endpoint: http://127.0.0.1:8000/health

The exact available features depend on the configured application
environment and user role.

8. Testing

Run the automated test suite from the project root:

python -m pytest -v

The tests cover application endpoints, authentication behavior, and
role-based access restrictions.

9. Dataset and Evaluation

The project uses a resume dataset containing resume text and associated
job-category labels. The dataset is used to train and evaluate multiple
machine learning classification algorithms.

The selected model is a calibrated Linear Support Vector Machine (Linear
SVM) using TF-IDF text features.

Final Test Results:
- Test Accuracy: 72.43%
- Test Macro F1-score: 68.00%
- Test Weighted F1-score: 71.55%
- Number of Test Samples: 497

The model was evaluated on a held-out test set. Accuracy measures the
proportion of correctly classified resumes. Macro F1 calculates the
F1-score for each category and averages them equally, while Weighted F1
accounts for the number of samples in each category.

Additional evaluation results, including per-category classification
metrics and the confusion matrix, are available in the reports/
directory.

The model assists with initial resume screening and should not replace
human review or be used as the sole basis for recruitment decisions.

10. Security and Limitations

ResumeIQ includes authentication, role-based access control, and
validation for resume uploads. These measures support safer application
use but do not, by themselves, establish that the system has undergone a
comprehensive security audit.

Predictions may be inaccurate, particularly for resumes that are
incomplete, unusually formatted, or outside the categories represented in
the training dataset. Confidence scores should be interpreted cautiously,
and recruitment decisions should remain subject to human review.

Before deploying the application publicly, configure secure secrets,
production server settings, database protection, and appropriate handling
of personal information.

11. Future Improvements

Potential future improvements include:

* Deploying the application to a production hosting environment.
* Adding company-specific resume submission and application tracking.
* Expanding and improving the training dataset.
* Enhancing model evaluation and explainability.
* Adding further security and deployment automation.

12. Academic Project

Project Title: NLP-Based Resume (CV) Classification and Web-Based Human
Resources Management System

Course: WEX 428 — Graduation Project / Internship

Institution: OSTİM Technical University

⸻

This project is an academic prototype intended to demonstrate machine
learning-based resume classification and web-based HR management.
