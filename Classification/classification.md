# Classification

## Random Forest Classifier

```
# build random forest classifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)    
rfc_predict = rfc.predict(X_test)

scores_rfc = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')
mean_rfc = ("Mean AUC Score - Random Forest: ", scores_rfc.mean())

print(confusion_matrix(y_test, rfc_predict))
print('\n')
print(classification_report(y_test, rfc_predict))

scores_rfc = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')
print(scores_rfc)

print('\n')

print("Mean AUC Score - Random Forest: ", scores_rfc.mean())

feature_importances = pd.DataFrame(rfc.feature_importances_,
                               index = X_train.columns,
                                columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
```
