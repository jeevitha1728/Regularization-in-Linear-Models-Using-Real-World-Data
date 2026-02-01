# Regularization-in-Linear-Models-Using-Real-World-Data

Objective
This assignment focuses on controlling overfitting using regularization
techniques. You will learn how Ridge, Lasso, and Elastic Net modify linear
regression behavior and why regularization is essential in real-world datasets.

Dataset:
The dataset contains 500 samples and 6 features.

Train–Test Split
The dataset is split into training and testing sets:
75% training data
25% testing data

Feature Scaling:

Standardization is applied using StandardScaler.

Features are scaled to zero mean and unit variance.
This step is essential for regularization, as Ridge, Lasso, and Elastic Net penalize coefficient.
Without scaling, features with larger values would dominate the regularization term.

Multiple Linear Regression :
multiple linear regression model is trained without regularization.
The model minimizes only the mean squared error (MSE).
It serves as a baseline for comparison with regularized models.


alpha controls the strength of regularization.
Larger alpha values impose stronger penalties on model coefficients.
Multiple values are tested to observe how increasing regularization affects performance.

ridge observation:
Coefficients are shruk gradually as alpha increases.
No coefficient becomes exactly zero.
Training error increases with alpha.

lasso observation:
Some coefficients become exactly zero as alpha increases.
Lasso removes irrelevant features from the model.

elastic net observation :
Combines shrinkage from Ridge and sparsity from Lasso.
More stable when features are correlated.

Conclusion: 
Regularization reduces overfitting.
Ridge shrinks coefficients.
Lasso performs feature elimination.
Elastic Net provides a balanced approach.
