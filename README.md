# 🏡 House Price Prediction App

This project is a **machine learning-powered web application** that predicts the **selling price of a residential property** based on user inputs. Built using **Streamlit**, it provides a clean, interactive interface and is backed by a trained **Random Forest Regression model** for accurate predictions.

## To run the app
https://amira-aladra-house-prediction-app.streamlit.app/

---

## 🚀 Project Overview

This app allows users to:
- Input key features about a house (e.g., size, age, bedrooms, location type, taxes)
- Instantly receive a price prediction based on a trained model
- View interactive visualizations derived from real estate data
- Understand which features most influence house prices

It’s ideal for:
- Real estate analysts
- Property investors
- Data science learners building end-to-end projects

---

## 📦 Features

- 🔍 **Feature Engineering**: Created new variables like `property_age`, `popular` homes, and `recession` timing.
- 📈 **Data Visualizations**: Auto-generated correlation heatmap and boxplot of price by property type.
- 🤖 **ML Model**: Trained using a Random Forest Regressor for robust predictions.
- 🌐 **Streamlit Web App**: Easy-to-use interface for real-time prediction and visualization.

---

## 🧠 Why Random Forest?

We used a **Random Forest Regressor** because:

- ✅ It handles both linear and non-linear relationships well
- ✅ Works with mixed feature types (categorical + numerical)
- ✅ Is robust against overfitting (due to averaging across trees)
- ✅ Provides good performance even with minimal hyperparameter tuning

**Criterion used**: `absolute_error` for better interpretability and robustness against outliers (compared to MSE).

## 🧪Machine Learning Details
Model	RandomForestRegressor (200 trees, absolute_error criterion)
Target Variable:	price
Feature Set	Engineered + dummy encoded
Train/Test Split	80/20
Evaluation Metric	Mean Absolute Error (MAE)

## 📊 Dataset Description

The dataset is a CSV file containing historical housing data, including:
- `beds`, `baths`, `sqft`, `lot_size`
- `year_built`, `year_sold`
- `property_type`: e.g., Condo, Bungalow
- `basement`, `property_tax`, `insurance`, `price`

Preprocessing includes:
- Cleaning null values
- Dropping outliers (`lot_size > 500,000`)
- Type conversions (`basement` to `int`)
- Generating engineered features (`popular`, `recession`, etc.)

---

## 🖥️ Tech Stack

| Layer | Tools Used |
|-------|------------|
| 💻 Frontend | [Streamlit](https://streamlit.io) |
| 📊 Visualization | Matplotlib, Seaborn |
| 🧠 Machine Learning | Scikit-learn (Random Forest Regressor) |
| 🐍 Backend/Logic | Python, Pandas |
| 📁 Packaging | Pickle |

---

## 📊 Sample Visualizations

- 📈 **Feature Correlation Heatmap**
- 📦 **Price Distribution by Property Type**

These plots are auto-generated from the dataset and included within the app.

---
## Project Structure
    Copy
    Edit
    ├── app.py                        # Streamlit application
    ├── models/
    │   └── RFmodel.pkl              # Trained Random Forest model
    ├── data/
    │   └── raw/real_estate.csv      # Source dataset
    ├── src/
    │   ├── data/
    │   │   └── data_processing.py
    │   ├── features/
    │   │   └── features_eng.py
    │   ├── models/
    │   │   ├── train_model.py
    │   │   └── predict_model.py
    │   └── visualization/
    │       └── visualize.py
    ├── heatmap.png                  # Generated correlation heatmap
    ├── boxplot.png                  # Generated boxplot by property type
    ├── README.md
    └── requirements.txt

    
📬 Contact
## 🚀 How to Run the App

### 1. Clone the repository
```bash
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
