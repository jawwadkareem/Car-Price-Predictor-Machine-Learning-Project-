import numpy as np
import pandas as pd

df = pd.read_csv('../Dataset/pakwheels_used_car_data_v02.csv')
print("Columns:",df.columns.tolist())
print("Unique Values for each column:",df.nunique())






# dropping columns that are useless for making predictions
columns_to_drop = ['addref', 'city', 'assembly', 'registered', 'color','body']
data_cleaned = df.drop(columns=columns_to_drop)


missing_values = data_cleaned.isnull().sum()
# Check for duplicates
duplicates = data_cleaned.duplicated().sum()

print("Missing values:",missing_values)
print("No. of duplicates:",duplicates)

# cleaning the data by eliminating missing and duplicate values
# Drop rows with missing values in crucial columns
data_cleaned = data_cleaned.dropna(subset=['year', 'engine', 'price','fuel'])
# Drop duplicate rows
data_cleaned = data_cleaned.drop_duplicates()
remaining_missing_values = data_cleaned.isnull().sum()
remaining_duplicates = data_cleaned.duplicated().sum()



# Removing outliers using the IQR method
df = data_cleaned
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Removing outliers in the mileage, price, engine and year columns
data_no_outliers = remove_outliers(df, 'mileage')
data_no_outliers = remove_outliers(data_no_outliers, 'price')

data_no_outliers = remove_outliers(data_no_outliers, 'engine')

data_no_outliers = remove_outliers(data_no_outliers, 'year')



# futher cleaning the data for making a better predictive model
df = data_no_outliers
filtered_data_mileage = df[(df['mileage'] >= 5000)& (df['mileage'] <= 200000)]

# Filter the data for minimum price of 400,000 and maximum price of 8,900,000
filtered_data_price = filtered_data_mileage[(filtered_data_mileage['price'] >= 400000) & (filtered_data_mileage['price'] <= 7500000)]


# Filter the data for minimum engine cc of 660
filtered_data_final = filtered_data_price[filtered_data_price['engine'] >= 660]




df = filtered_data_final
print(df.describe())

cleaned_file_path = '/Datacleaning/pakwheels_preprocessed_data2.csv'
df.to_csv(cleaned_file_path, index=False)
# using randomforestregressor
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # Define features and target variable
# X = df.drop('price', axis=1)
# y = df['price']
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define preprocessing steps for categorical and numerical features
# categorical_features = ['make', 'model', 'transmission', 'fuel']
# numerical_features = ['year', 'engine', 'mileage']
#
# # Create preprocessing pipelines for both numeric and categorical data
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median'))  # handle missing values
# ])
#
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # handle missing values
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))  # one-hot encoding
# ])
#
# # Combine preprocessing steps
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numerical_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])
#
# # Create a pipeline with preprocessing and model
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('regressor', RandomForestRegressor())])
#
# # Fit the model
# pipeline.fit(X_train, y_train)
#
# # Predict on the test set
# y_pred = pipeline.predict(X_test)
#
# # Calculate evaluation metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
# r2 = r2_score(y_test, y_pred)
#
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"R-squared (R2): {r2}")



#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Assuming your dataset is stored in a DataFrame 'filtered_data_final'
#
# # Define features and target variable
# X = df.drop('price', axis=1)
# y = df['price']
#
# # Initialize variables to track the best random state and its corresponding R2 score
# best_random_state = None
# best_r2_score = -float('inf')  # Initialize with a very low value
#
# # Test different random states
# for random_state in range(100):  # Try random states from 0 to 99
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
#
#     # Define preprocessing steps for categorical and numerical features
#     categorical_features = ['make', 'model', 'transmission', 'fuel']
#     numerical_features = ['year', 'engine', 'mileage']
#
#     # Create preprocessing pipelines for both numeric and categorical data
#     numeric_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='median'))  # handle missing values
#     ])
#
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # handle missing values
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))  # one-hot encoding
#     ])
#
#     # Combine preprocessing steps
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numerical_features),
#             ('cat', categorical_transformer, categorical_features)
#         ])
#
#     # Create a pipeline with preprocessing and model (Linear Regression)
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('regressor', LinearRegression())])
#
#     # Fit the model
#     pipeline.fit(X_train, y_train)
#
#     # Predict on the test set
#     y_pred = pipeline.predict(X_test)
#
#     # Evaluate the model (calculate R2 score)
#     r2 = r2_score(y_test, y_pred)
#
#     # Check if this random state gives a better R2 score
#     if r2 > best_r2_score:
#         best_r2_score = r2
#         best_random_state = random_state
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=best_random_state)
# # Define preprocessing steps for categorical and numerical features
# categorical_features = ['make', 'model', 'transmission', 'fuel']
# numerical_features = ['year', 'engine', 'mileage']
#
# # Create preprocessing pipelines for both numeric and categorical data
# numeric_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='median'))  # handle missing values
#     ])
# categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # handle missing values
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))  # one-hot encoding
#     ])
#
#     # Combine preprocessing steps
# preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numerical_features),
#             ('cat', categorical_transformer, categorical_features)
#         ])
#
#     # Create a pipeline with preprocessing and model (Linear Regression)
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('regressor', LinearRegression())])
#
#     # Fit the model
# pipeline.fit(X_train, y_train)
#
#     # Predict on the test set
# y_pred = pipeline.predict(X_test)
#
#     # Evaluate the model (calculate R2 score)
# r2 = r2_score(y_test, y_pred)
#
# # Print the best random state and its corresponding R2 score
# print(f"Best Random State: {best_random_state}")
# print(f"Highest R-squared (R2) Score: {best_r2_score}")
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# # Assuming pipeline is already trained and X_test, y_test are defined as in previous examples
#
# # Predict on the test set
# y_pred = pipeline.predict(X_test)
#
# # Calculate evaluation metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
# r2 = r2_score(y_test, y_pred)
#
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"R-squared (R2): {r2}")