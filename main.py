import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = 'train.csv'
df = pd.read_csv(file_path)
# Display the first few rows and column names
print(df.head())
print(df.columns)

# Check for 'SalePrice' column
if 'SalePrice' not in df.columns:
    print("Error: 'SalePrice' column not found in the dataset.")
else:
    # Data Preprocessing
    # Handle missing values
    df.ffill(inplace=True)

    # Define features and target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # Separate numerical and categorical features
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object]).columns.tolist()

    # Preprocessing pipelines for numerical and categorical data
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline for model training
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R2 Score: {r2}')

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.show()

    # Box Plot for LotArea vs HouseStyle
    plt.figure(figsize=(12, 8))
    sns.barplot(x='HouseStyle', y='LotArea', data=df)
    plt.title('LotArea vs HouseStyle - Box Plot')
    plt.xlabel('HouseStyle')
    plt.ylabel('LotArea')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.lineplot(x='YrSold', y='YearBuilt', data=df)
    plt.title('YrSold vs YearBuilt - Line Plot')
    plt.xlabel('Year Sold')
    plt.ylabel('Year Built')
    plt.show()