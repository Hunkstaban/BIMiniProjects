# MP3 Questions
### Peter:
#### Which ML model did you choose to apply and why?
For the classification of employee attrition, I applied two supervised machine learning methods: Decision Tree and Random Forest classifiers.

Decision Tree was chosen for its interpretability (by being able to see a visual representation) and ability to handle both categorical and numerical data without the need for normalization or outlier removal.
Random Forest was chosen as I wasn't satisfied with the accuracy score from the Decision Tree model, and the Random Forest could potentially provide that while being more robust by combining the predictions of multiple decision trees.

#### How accurate is your solution of prediction? Explain the meaning of the quality measures.
- The Decision Tree model achieved a test accuracy of approximately 0.79 and a cross-validated accuracy of 0.78.
- The Random Forest model achieved a higher test accuracy of approximately 0.85 and a cross-validated accuracy of 0.84.

Accuracy measures the proportion of correct predictions out of all predictions made.
I also used a confusion matrix and a classification report (which includes precision, recall, and F1-score) to evaluate the models. 
Explanation of the different values:
- Precision: The proportion of positive identifications that were actually correct.
- Recall: The proportion of actual positives that were correctly identified.
- F1-score: The harmonic mean of precision and recall, providing a balance between the two.

These metrics help us understand overall accuracy and how well the model identifies employees who are likely to leave.

#### Which are the most decisive factors for quitting a job? Why do people quit their job?
Based on the feature importance analysis from both the Decision Tree and Random Forest models, the most decisive factors for predicting employee attrition are:
- Monthly Income
- OverTime
- Age
- Total Working Years
- Distance From Home
- Job Level and Job Role

From this it can be interperted that people are most likely to quit their jobs due to low pay, high job demands (such as overtime), long commutes, and possibly being early in their careers or in certain roles/levels that may not meet their expectations or needs - or possibly a combination of these.

#### What could be done for further improvement of the accuracy of the models?
Potentially some feature engineering where either creating new features or combining existing ones could better capture patterns in the data.

#### Which were the challenges in the project development?
There was no strong linear correlations with attrition, making it necessary to rely on the models to capture the interactions. This similarly made it hard to decide which features to potentially exclude, making the visual representaiton of the decision tree huge and hard to interpret. 