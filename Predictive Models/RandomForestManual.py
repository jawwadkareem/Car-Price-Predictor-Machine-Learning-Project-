from Scaling_Encoding import scaling_and_encoding as sae
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import joblib



from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_features=self.max_features, random_state=np.random.randint(0, 10000))
            X_sample, y_sample = resample(X, y, random_state=np.random.randint(0, 10000))#random sampling is done here
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])#taking the mean of predictions made by all decision trees 
        return np.mean(tree_preds, axis=0)

#Training the model
pipeline = Pipeline(steps=[('preprocessor', sae.preprocessor),
                           ('regressor', CustomRandomForestRegressor(n_estimators=100, max_features='sqrt',random_state=98))])

pipeline.fit(sae.X_train, sae.y_train)

joblib.dump(pipeline, 'car_price_prediction_random_forest_manual1.pkl')

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
