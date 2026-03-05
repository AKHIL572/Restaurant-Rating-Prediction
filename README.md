# 🍽 Restaurant Rating Prediction System

A **Machine Learning powered analytics system** that predicts the **aggregate rating of a restaurant** based on location, cost, cuisines, and service features.

This project demonstrates a **complete end-to-end Data Science workflow**, including:

- Data Understanding
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Modeling
- Hyperparameter Tuning
- Model Evaluation
- Model Deployment using Streamlit

---

# 📊 Problem Statement

Restaurant platforms contain thousands of restaurants with attributes such as:

- Location
- Cuisine
- Price range
- Delivery services
- Customer votes

The goal of this project is to **predict the restaurant’s aggregate rating** using machine learning techniques so businesses can understand the factors that influence customer satisfaction.

---

# 🧠 Machine Learning Workflow

The project follows a structured ML pipeline:

1. Data Understanding  
2. Exploratory Data Analysis  
3. Feature Engineering  
4. Data Preprocessing  
5. Model Training  
6. Hyperparameter Tuning  
7. Model Evaluation  
8. Model Deployment

---

# 📂 Project Structure

```
Restaurant_Rating_Prediction
│
├── Dataset/
│ ├── Res_dataset.csv
│ ├── cleaned_dataset.csv
│
├── Models/
│ ├── restaurant_rating_model.pkl
│ └── feature_columns.pkl
│
├── Notebook/
│ ├── 1_data_understanding.ipynb
│ ├── 2_eda.ipynb
│ ├── 3_preprocessing_&_modelling.ipynb
│ └── 4_business_insights.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Information

The dataset contains **9551 restaurant records** with attributes such as:

| Feature | Description |
|------|-------------|
| Restaurant Name | Name of the restaurant |
| Country Code | Country identifier |
| City | City location |
| Latitude / Longitude | Geographic coordinates |
| Cuisines | Types of cuisines offered |
| Average Cost for Two | Approximate meal cost |
| Price Range | Cost category |
| Has Table Booking | Reservation availability |
| Has Online Delivery | Delivery availability |
| Votes | Number of customer votes |
| Aggregate Rating | Target variable |

---

# 🔎 Exploratory Data Analysis

EDA was performed to understand patterns in the data:

- Rating distribution analysis
- Cuisine popularity analysis
- Price range vs rating relationship
- Geographic restaurant distribution
- Feature correlation analysis
- Outlier detection

These insights helped guide **feature engineering and model selection**.

---

# ⚙ Feature Engineering

The following engineered features were created:

- **Cuisine_Count** → Number of cuisines offered
- **Log_Votes** → Log transformation of votes
- **Log_Average_Cost** → Log transformation of cost
- **Cuisine Encoding** → Top cuisines converted into binary features
- **Cost_Category** → Derived cost class (Low / Medium / High / Premium)

---

# 🤖 Models Used

Several models were trained and compared:

| Model | Purpose |
|------|--------|
| Linear Regression | Baseline model |
| Decision Tree | Non-linear modeling |
| Random Forest | Ensemble model |
| Gradient Boosting | Final optimized model |

---

# 🏆 Best Model

**Gradient Boosting Regressor**

Performance metrics:

| Metric | Score |
|------|------|
| R² Score | ~0.67 |
| RMSE | ~0.32 |
| MAE | ~0.23 |

After hyperparameter tuning, **Gradient Boosting provided the best performance**.

---

# 🚀 Streamlit Web Application

A **Streamlit app** was built to allow users to interact with the model.

Users can input:

- Country
- City
- Price Range
- Average Cost for Two
- Delivery Availability
- Table Booking
- Number of Votes
- Cuisines

The model predicts the **expected restaurant rating**.

---

# 🖥 How to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/restaurant-rating-prediction.git
cd restaurant-rating-prediction
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
📈 Example Prediction

Example Input:
City → Bangalore
Price Range → Medium
Cost for Two → 500
Online Delivery → Yes
Votes → 100
Cuisine → North Indian, Chinese
Output:
Predicted Rating: 4.1 / 5

📊 Business Insights
Important findings from the analysis:
- Number of Votes strongly influences rating reliability
- City and location affect restaurant popularity
- Cuisine types influence customer preference
- Price category correlates with perceived quality
These insights help restaurant owners and food delivery platforms make data-driven decisions.

🛠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

👨‍💻 Author

Akhil T V
Aspiring Data Scientist / Data Analyst

This project demonstrates my ability to build end-to-end machine learning solutions from data exploration to deployment.
