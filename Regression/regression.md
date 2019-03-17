# Regression Models

Linear Regression
Decision Tree
Random Forest
Extra Trees
Support Vector Machines
Lasso Regression
Ridge Regression

# Linear Regression


```
# linear regression
lm = LinearRegression()
model_lm = lm.fit(X_train,y_train)
pred_lm = lm.predict(X_test)
result_lm = model_lm.score(X_test, y_test)

# cross-validation
cv_scores_lm = cross_val_score(lm, X, y, cv=5, scoring='neg_mean_squared_log_error')
lm_rmsle = np.sqrt(np.abs(cv_scores_lm.mean()))

# linear model scoring output
print("=== Linear Regression ===")
print("R2: " + str(result_lm))
print("RMSLE: " + str(rmsle(pred_lm, y_test)))
print("RMSE: " + str(rmse(pred_lm, y_test)))
print("Mean RMSLE Score: ", lm_rmsle.mean())
print("\n")
```

# Random Forest Regression

```
# Random Forest Regression
rfr = RandomForestRegressor()
model_rfr = rfr.fit(X_train,y_train)
pred_rfr = rfr.predict(X_test)
result_rfr = model_rfr.score(X_test, y_test)
cv_scores_rfr = cross_val_score(rfr, X, y, cv=5, scoring='neg_mean_squared_log_error')
rfr_rmsle = np.sqrt(np.abs(cv_scores_rfr.mean()))

print("=== Random Forest Regression ===")
print("R2: " + str(result_rfr))
print("RMSLE: " + str(rmsle(pred_rfr, y_test)))
print("RMSE: " + str(rmse(pred_rfr, y_test)))
print("Mean RMSLE Score: ", rfr_rmsle.mean())
print("\n")
```

Hyperparameter tuning


# Scoring Metrics

R-Squared
Root Mean Squared Logarithmic Error (RMSLE)
