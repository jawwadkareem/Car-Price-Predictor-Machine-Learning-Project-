from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from Scaling_Encoding import scaling_and_encoding

pipeline = Pipeline(steps=[('preprocessor', scaling_and_encoding.preprocessor),
                           ('regressor', RandomForestRegressor(random_state=48))])

pipeline.fit(scaling_and_encoding.X_train, scaling_and_encoding.y_train)
# joblib.dump(pipeline, 'car_price_prediction_random_forest_library.pkl')
y_pred = pipeline.predict(scaling_and_encoding.X_test)

mae = mean_absolute_error(scaling_and_encoding.y_test, y_pred)
r2 = r2_score(scaling_and_encoding.y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")