# House Prices Prediction Project

## Overview
This project aims to predict house prices using various features of houses. The dataset used in this project is the `train.csv` file from the Kaggle competition "House Prices - Advanced Regression Techniques". The goal is to build a regression model to predict the sale price of houses based on their features. The project also includes data preprocessing, exploratory data analysis (EDA), and model evaluation.

## Project Description
This project involves several key steps:
1. **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling numerical features.
2. **Exploratory Data Analysis (EDA)**: Visualizing the data to understand relationships between features and the target variable.
3. **Model Building**: Creating a regression model using Linear Regression.
4. **Model Evaluation**: Evaluating the model's performance using various metrics.
5. **Visualization**: Plotting various relationships and model predictions.

## Tools and Libraries
- **Python**: The programming language used for the project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For machine learning, including data preprocessing, model building, and evaluation.

## Project Workflow

### 1. Data Preprocessing
- **Handling Missing Values**: Using forward fill to handle missing values.
- **Encoding Categorical Variables**: Using `OneHotEncoder` to convert categorical variables into a format that can be provided to ML algorithms.
- **Scaling Numerical Variables**: Using `StandardScaler` to standardize the features.

### 2. Exploratory Data Analysis (EDA)
- **Box Plot for `LotArea` vs. `HouseStyle`**: Visualizes the distribution of lot area across different house styles.
- **Line Plot for `YrSold` vs. `YearBuilt`**: Shows the relationship between the year sold and the year built.

### 3. Model Building
- **Linear Regression**: Creating a regression model to predict house prices.

### 4. Model Evaluation
- **Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) to evaluate the model.

### 5. Visualization
- **Actual vs Predicted Prices**: Scatter plot to visualize the performance of the model.
- **Additional Visualizations**: Box plots and line plots to explore relationships in the data.
