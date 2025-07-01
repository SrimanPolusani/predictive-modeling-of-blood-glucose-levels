# Import statements
import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.robust.norms import HuberT
from eda import m1_df, continuous_vars
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create a variable for response variable
Y = m1_df['HbA1c_Box_Cox']

# Create a variable for predictor candidates
# Change the object type of factors from int to str so that Statsmodels module as recognize that it as a category.
X_candidates = m1_df[['Age', 'Gender', 'Race', 'BMI_Box_Cox', 'BP_Box_Cox', 'HS_C_Box_Cox', 'TC_to_HDL_Box_Cox']].copy()
# X_candidates = m1_df[['Age', 'Gender', 'Race', 'BMI_Box_Cox', 'BP_Box_Cox']].copy()
X_candidates['Gender'] = X_candidates['Gender'].map({1.0: 'Male', 2.0: 'Female'}).astype('category')

race_mappping = {
    1.0: 'Mexican American',
    2.0: 'Other Hispanic',
    3.0: 'Non-Hispanic White',
    4.0: 'Non-Hispanic Black',
    6.0: 'Non-Hispanic Asian',
    7.0: 'Other Race - Including Multi-Racial'
}

X_candidates['Race'] = X_candidates['Race'].map(race_mappping).astype('category')

model_data = pd.concat([X_candidates, Y], axis=1)

print("First 15 obs of the model_data")
print(tabulate(model_data.head(15), headers='keys', tablefmt='psql'))

# Create a variable for storing response variable name
Y_name = 'HbA1c_Box_Cox'

# Stepwise Regression Function
def forward_selection(data, response, alpha):
    """Performs forward selection"""
    initial_predictors = data.columns.tolist()
    initial_predictors.remove(response)
    selected_predictors = []

    print(f"----------Starting Forward Selection at aplha = {alpha}----------")

    while True:
        remaining_predictors = list(set(initial_predictors) - set(selected_predictors))
        best_pval = 1.0  # Initialize it will p value greater than alpha
        feature_to_add = None
        for predictor in remaining_predictors:
            current_formula = f"{response} ~ {" + ".join(selected_predictors + [predictor])}"
            print(current_formula)
            try:
                model = smf.rlm(current_formula, data=model_data, M=HuberT()).fit()
                related_coeffs = [coeff for coeff in model.pvalues.index if predictor in coeff]
                if not related_coeffs:
                    print("There are no related coefficients. Continuing next loop")
                    continue
                print(f"related coeffs: {related_coeffs}")
                print(f"pvalues of  the model:\n {model.pvalues}")
                current_pval = min(model.pvalues[coeff] for coeff in related_coeffs)

                if current_pval < best_pval:
                    best_pval = current_pval
                    print(f"current best p value is: {best_pval:.4f}\n")
                    feature_to_add = predictor
                else:
                    print(f"Haven't changed current best p value as {current_pval:.4f} >= {best_pval:.4f}\n")

            except Exception as e:
                print(f"Error fitting the {current_formula}. Error: {e}")
                continue

        if feature_to_add and best_pval < alpha:
            selected_predictors.append(feature_to_add)
            print(f"Adding feature: {feature_to_add}, p value: {best_pval:.4f}\n")
        else:
            if feature_to_add is None:
                print("No more features to add\n")
            elif best_pval >= alpha:
                print(f"{feature_to_add} is not added as it is less than {alpha}. p-value: {best_pval:.4f}\n")
            break
    print("-------------Forward Selection Final Model Formula-------------")
    final_formula = f"{response} ~ {" + ".join(selected_predictors)}" if selected_predictors else f"{response} ~ 1"
    print(f"Selected Model Formula: {final_formula}\n")
    final_model = smf.rlm(final_formula, data, M=HuberT()).fit()
    print(final_model.summary())

    return selected_predictors, final_model

# Perform forward selection
selected_features, selected_model = forward_selection(data=model_data, response=Y_name, alpha=0.05)

# Examine Residual plots
residuals = selected_model.resid
fitted_values = selected_model.fittedvalues

# Visualize Residuals vs Fitted values
plt.figure(figsize=(12, 8))
sns.scatterplot(y=residuals, x=fitted_values)
plt.axhline(y=0, color='red', linestyle='-.')
plt.title("Residuals vs Fitted values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()
# Conclusion: No non-linear patterns observed, Mild Heteroscedasticity and presence of outliers

# Visualize histogram of residuals
plt.figure(figsize=(12, 8))
sns.histplot(x=residuals, kde=True, bins=40)
plt.title('Histogram of residuals')
plt.xlabel("Residuals")
plt.ylabel("Count")
plt.grid(True)
plt.show()
# Conclusion: Although the curve looks symmetrical, it is skewed to the left

# Plot Q-Q plot of residuals
plt.figure(figsize=(10, 10))
sm.qqplot(residuals, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()
# Conclusion: Major deviation in the points at the left from red line, indicating negative skewness

# Visualize Residuals vs Predictors
con_predictors = ['HS_C_Box_Cox', 'TC_to_HDL_Box_Cox']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
for i, predictor in enumerate(con_predictors):
    sns.scatterplot(y=residuals, x=model_data[predictor], ax=axes[i])
    axes[i].axhline(y=0, color='red', linestyle='-.')
    axes[i].set_title("Residuals vs Predictors")
    axes[i].set_xlabel(f"{predictor}")
    axes[i].set_ylabel("Residuals")
    axes[i].grid(True)

plt.tight_layout()
plt.show()
# Conclusion: no non-linear pattern and no major predictor specific heteroscedasticity


# 4a. Prepare data
data = model_data.copy()
features = selected_features # From your earlier forward selection
response = Y_name

# 4b. Split into train (80%) / test (20%)
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# 4c. Refit Robust Linear Model on training set
eval_formula = f"{response} ~ {' + '.join(features)}"
train_rlm = smf.rlm(eval_formula, data=train_df, M=HuberT()).fit()

# 4d. Predict on test set
y_true = test_df[response]
y_pred = train_rlm.predict(test_df)

# 4e. Compute accuracy metrics
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
# Note: R-squared is calculated but less standard for RLM evaluation
r2   = r2_score(y_true, y_pred)

print("\nRLM Test Set Performance:")
print(f"  RMSE: {rmse:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"    R²: {r2:.4f}") # R-squared calculated by scikit-learn

print("\n" + "="*60)
print("="*60 + "\n")

# --- Plot 1: Actual vs. Predicted Values ---
plt.figure(figsize=(10, 10)) # Make it square
# Use y_true and y_pred calculated in the previous block
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=50) # s adjusts point size

# Add a line for perfect prediction (y=x)
# Use min/max from y_true and y_pred
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

plot_buffer = (max_val - min_val) * 0.05
min_val -= plot_buffer
max_val += plot_buffer

plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')

plt.title('Test Set: Actual vs. Predicted Values', fontsize=16)
plt.xlabel(f'Actual {response}', fontsize=12)
plt.ylabel(f'Predicted {response}', fontsize=12)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# --- Plot 2: Histogram of Prediction Errors (Test Set Residuals) ---
# Calculate residuals using y_true and y_pred
test_residuals = y_true - y_pred

plt.figure(figsize=(12, 6))
sns.histplot(test_residuals, kde=True, bins=40)
plt.axvline(0, color='red', linestyle='--') # Line at zero error
plt.title('Histogram of Prediction Errors (Test Set Residuals)', fontsize=16)
plt.xlabel(f'Prediction Error (Actual - Predicted {response})', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y')
plt.show()