import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import  r2_score,mean_absolute_error
import scaling_and_encoding as sae
import joblib
import pandas as pd


class GradientDescentRegressor(BaseEstimator, RegressorMixin):
    def _init_(self, learning_rate=0.01, num_iterations=10000, regularization='l2', alpha=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.alpha = alpha  # Regularization strength

    def fit(self, X, y):
        if hasattr(X, 'toarray'):
            X = X.toarray()

        X = np.column_stack((np.ones(len(X)), X))
        self.theta_ = np.zeros((X.shape[1], 1))
        m = len(y)
        y = y.values.reshape(-1, 1)

        for i in range(self.num_iterations):
            predictions = np.dot(X, self.theta_)
            errors = predictions - y
            gradient = (1 / m) * np.dot(X.T, errors)

            if self.regularization == 'l2':
                gradient += (self.alpha / m) * self.theta_

            self.theta_ -= self.learning_rate * gradient

        return self

    def predict(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()

        X = np.column_stack((np.ones(len(X)), X))
        y_pred = np.dot(X, self.theta_)
        return y_pred

# X_train, X_test, y_train, y_test = train_test_split(sae.X, sae.y, test_size=0.2, random_state=78)
# pipeline = Pipeline(steps=[
#     ('preprocessor', sae.preprocessor),
#     ('regressor', GradientDescentRegressor(learning_rate=0.01, num_iterations=10000, regularization='l2', alpha=0.01))
# ])
#
# pipeline.fit(X_train, y_train)
# joblib.dump(pipeline, 'car_price_prediction_linear_regression_manual.pkl')

# Predict on the test set
# y_pred = pipeline.predict(X_test)


pipeline_loaded = joblib.load('car_price_prediction_linear_regression_manual.pkl')
# Predict on the test set
y_pred = pipeline_loaded.predict(sae.X_test)


# pipeline_loaded = joblib.load('car_price_prediction_random_forest_manual.pkl')
# y_pred = pipeline_loaded.predict(sae.X_test)
mae = mean_absolute_error(sae.y_test, y_pred)
r2 = r2_score(sae.y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")


sample_data = {
    'make': ['Toyota'],
    'model': ['Yaris'],
    'year': [2021.0],
    'engine': [1300.0],
    'transmission': ['Automatic'],
    'fuel': ['Petrol'],
    'mileage': [25000]
}

# Convert sample input into a DataFrame
sample_df = pd.DataFrame(sample_data)

# Use the trained pipeline to predict the price for the sample input
predicted_price = pipeline_loaded.predict(sample_df)

print(f"Predicted Price: {predicted_price[0]}")