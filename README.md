# Predictive Modeling of Glycohemoglobin (HbA1c) using NHANES Data

## 📖 Project Overview

This project aims to predict Glycohemoglobin (HbA1c) levels, a key indicator of long-term blood sugar control, using demographic, body measurement, and biomarker data from the National Health and Nutrition Examination Survey (NHANES). The analysis involves an end-to-end data science pipeline, including data integration, exploratory data analysis, feature engineering, robust statistical modeling, and model evaluation.

The primary goal is to identify the most significant predictors of HbA1c and build a reliable model that can handle outliers and non-normal distributions commonly found in biomedical data.

---

## 📂 Project Structure

The repository is organized as follows:

```
.
├── data_files/
│   ├── DEMO_L.xpt
│   ├── BMX_L.xpt
│   ├── GHB_L.xpt
│   ├── GLU_L.xpt
│   ├── BPXO_L.xpt
│   ├── INS_L.xpt
│   ├── HSCRP_L.xpt
│   ├── TCHOL_L.xpt
│   └── HDL_L.xpt
├── scripts/
│   ├── box_cox.py
│   ├── eda.py
│   └── feature_selection.py
└── README.md
```

* `data_files/`: Contains the raw `.xpt` data files from the NHANES survey.
* `scripts/`: Contains the Python scripts for the analysis.
    * `box_cox.py`: A custom Python class for performing the Box-Cox transformation.
    * `eda.py`: Script for data loading, cleaning, merging, feature engineering, and exploratory data analysis.
    * `feature_selection.py`: Script for feature selection, model building, diagnostics, and evaluation.

---

## ⚙️ Methodology

The project follows a structured data science workflow, detailed below.

### 1. Data Ingestion & Preprocessing

* **Data Integration:** Eight separate NHANES datasets (`.xpt` files) were loaded using the `pandas` library. These datasets were merged on the respondent sequence number (`SEQN`) to create a single analytical dataset for 2,854 participants.
* **Feature Engineering:** Several new predictors were computed to capture important clinical relationships:
    * **Non-HDL Cholesterol:** $Total\ Cholesterol - HDL\ Cholesterol$
    * **TC-to-HDL Ratio:** $Total\ Cholesterol / HDL\ Cholesterol$
    * **Average Systolic Blood Pressure:** The mean of three separate systolic blood pressure readings.
* **Data Cleaning:** Rows with any missing values were identified and removed using `dropna()` to ensure model robustness.

### 2. Exploratory Data Analysis (EDA)

A comprehensive EDA was performed using `matplotlib` and `seaborn` to understand data distributions, identify relationships, and check assumptions.
* **Descriptive Statistics:** Summary statistics (mean, median, standard deviation), skewness, and kurtosis were calculated for all continuous variables.
* **Visualizations:**
    * **Histograms and Box Plots** were generated to assess the distribution and identify outliers in continuous variables.
    * **Count Plots** were used to visualize the distribution of categorical variables (Gender, Race).
    * A **Correlation Heatmap** was created to analyze multicollinearity between predictors.

### 3. Data Transformation (Box-Cox)

Several predictors and the response variable (HbA1c) exhibited significant skewness. To address this and meet the normality assumption for linear modeling, the **Box-Cox transformation** was applied.

The transformation is defined as:

$$
y(\lambda) =
\begin{cases}
\dfrac{y^\lambda - 1}{\lambda}, & \text{if } \lambda \neq 0 \\
\ln(y), & \text{if } \lambda = 0
\end{cases}
$$


A custom `BoxCox` class (`scripts/box_cox.py`) was implemented to find the optimal lambda ($\lambda$) value by maximizing the log-likelihood function over a specified range. This transformation was applied to the following variables:
* `BMI`
* `Insulin Resistance Ratio`
* `BP`
* `HS C-Reactive Protein`
* `Non_HDL_Chol`
* `TC_to_HDL_Ratio`
* `Poverty Ratio`
* `HbA1c` (Response Variable)

### 4. Feature Selection & Modeling

* **Model Choice:** A **Robust Linear Model (RLM)** from the `statsmodels` library was chosen. RLM is less sensitive to outliers than Ordinary Least Squares (OLS), making it suitable for this dataset. Huber's T norm was used for weighting.
* **Feature Selection:** A **forward stepwise selection** algorithm was implemented to identify the most statistically significant predictors. Starting with a null model, the algorithm iteratively adds the predictor with the lowest p-value until no remaining predictors have a p-value below the significance level ($\alpha = 0.05$).
* **Final Model:** The forward selection process identified 4 key predictors for the final model.

### 5. Model Evaluation & Diagnostics

* **Train-Test Split:** The dataset was split into training (80%) and testing (20%) sets using `scikit-learn` to evaluate the model's performance on unseen data.
* **Residual Analysis:** Diagnostic plots were generated to check the model's assumptions:
    * Residuals vs. Fitted Values Plot
    * Q-Q Plot of Residuals
    * Histogram of Residuals
* **Performance Metrics:** The model's predictive accuracy on the test set was quantified using:
    * Root Mean Squared Error (RMSE)
    * Mean Absolute Error (MAE)

---

## 📊 Results

The forward selection process identified the following variables as statistically significant predictors of Box-Cox transformed HbA1c:

* **Age**
* **Race**
* **HS_C_Box_Cox** (Transformed C-Reactive Protein)
* **TC_to_HDL_Box_Cox** (Transformed Total Cholesterol to HDL Ratio)

The final Robust Linear Model demonstrated strong predictive performance on the test set:
* **Root Mean Squared Error (RMSE):** 0.0014
* **Mean Absolute Error (MAE):** 0.0010

The Actual vs. Predicted plot shows a strong linear relationship, indicating the model's predictions are well-aligned with the true values.



---

## 🚀 How to Run

### Prerequisites

* Python 3.8+
* The required Python libraries listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *If a `requirements.txt` file is not available, install the packages manually:*
    ```bash
    pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tabulate
    ```
3.  **Update File Paths:** The script `scripts/eda.py` contains absolute file paths to the data. You must update these paths to reflect the location of the `data_files` directory on your local machine.

### Execution

Run the scripts from the root directory of the project in the following order:

1.  **Run the EDA and data processing script:** This will perform all preprocessing steps and generate visualizations. The final processed DataFrame is stored in memory.
    ```bash
    python scripts/eda.py
    ```
2.  **Run the feature selection and modeling script:** This script imports the processed DataFrame from `eda.py` to build and evaluate the final model.
    ```bash
    python scripts/feature_selection.py
    ```
The scripts will print model summaries and evaluation metrics to the console and display all graphical plots.
