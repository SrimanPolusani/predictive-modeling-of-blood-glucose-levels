# Import Pandas, Tabulate, Matplotlib and Seaborn
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.box_cox import BoxCox

# Set configuration for Seaborn and Matplotlib
sns.set_theme(style="darkgrid", palette="dark")
plt.rcParams['figure.figsize'] = (12,6)

# Read DEMO_L.xpt
demographics_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\DEMO_L.xpt")
demo_df_psql = tabulate(demographics_df.head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of DEMO_L.xpt (demographics table)")
print(demo_df_psql)

# Read BMX_L.xpt
body_mass_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\BMX_L.xpt")
body_mass_psql = tabulate(body_mass_df.head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of BMX_L.xpt (body mass table)")
print(body_mass_psql)

# Read GHB_L.xpt data
ghb_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\GHB_L.xpt", format='xport')
ghb_psql = tabulate(ghb_df.head(15), headers='keys', tablefmt='psql') 
print("\nFirst 15 obs of GHB_L.xpt (glycohemoglobin table)")
print(ghb_psql)

# Read GLU_L.xpt
glucose_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\GLU_L.xpt")
glucose_psql = tabulate(glucose_df.head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of GLU_L.xpt (glucose table)")
print(glucose_psql)

# Read BPXO_L.xpt
bp_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\BPXO_L.xpt")
bp_psql = tabulate(bp_df.head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of BPXO_L.xpt (BP table)")
print(bp_psql)

# Read INS_L.xpt data
insulin_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\INS_L.xpt", format='xport')
insulin_psql = tabulate(insulin_df.head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of INS_L.xpt (insulin table)")
print(insulin_psql)

# Read HSCRP_L.xpt data
inflammation_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\HSCRP_L.xpt", format='xport')
inflammation_psql = tabulate(inflammation_df.head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of HSCRP_L.xpt (inflammation table)")
print(inflammation_psql)

# Read TCHOL_L.xpt data
total_chol_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\TCHOL_L.xpt", format='xport')
total_chol_psql = tabulate(total_chol_df[['SEQN','LBDTCSI']].head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of TCHOL_L.xpt (total chol table)")
print(total_chol_psql)

# Read HDL_L.xpt data
hdl_df = pd.read_sas(r"C:\Users\srima\PycharmProjects\statistics_final_project\data_files\HDL_L.xpt", format='xport')
hdl_psql = tabulate(hdl_df[['SEQN','LBDHDDSI']].head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of TCHOL_L.xpt (total chol table)")
print(total_chol_psql)

# Compute predictors
total_chol_df['Non_HDL_Chol'] = total_chol_df['LBXTC'] - hdl_df['LBDHDD']
total_chol_df['TC_to_HDL_Ratio'] = total_chol_df['LBXTC'] / hdl_df['LBDHDD']
insulin_df['Insulin Resistance Ratio'] = insulin_df['LBDINSI'] * glucose_df['LBXGLU'] / 405

# Create datasets to merge
demo_subset = demographics_df[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR']]
bm_subset = body_mass_df[['SEQN', 'BMXBMI', 'BMXWAIST']]
insulin_subset = insulin_df[['SEQN', 'Insulin Resistance Ratio']]
inflammation_subset = inflammation_df[['SEQN', 'LBXHSCRP']]
chol_subset = total_chol_df[['SEQN', 'Non_HDL_Chol', 'TC_to_HDL_Ratio']]
HbA1c_subset = ghb_df[['SEQN', 'LBXGH']]

bp_subset = bp_df[['SEQN', 'BPXOSY1', 'BPXOSY2', 'BPXOSY3']].copy()
bp_subset['BPXOSY'] = bp_subset[
    ['BPXOSY1', 'BPXOSY2', 'BPXOSY3']
].mean(axis=1, skipna=False)
print("\nFirst 15 obs of Systolic BP table")
print(tabulate(bp_subset.head(15), headers='keys', tablefmt='psql'))

# Merged Datasets for the model
m1_df = pd.merge(demo_subset, bm_subset, on='SEQN', how='inner')
m1_df = pd.merge(m1_df, bp_subset[['SEQN', 'BPXOSY']], on='SEQN', how='inner')
m1_df = pd.merge(m1_df, inflammation_subset, on='SEQN', how='inner')
m1_df = pd.merge(m1_df, insulin_subset, on='SEQN', how='inner')
m1_df = pd.merge(m1_df, chol_subset, on='SEQN', how='inner')
m1_df = pd.merge(m1_df, HbA1c_subset, on='SEQN', how='inner')

# Rename the variables that are not descriptive
m1_df = m1_df.rename(columns={
    'SEQN': 'Seqn',
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Gender',
    'RIDRETH3': 'Race',
    'BMXBMI': 'BMI',
    'BPXOSY': 'BP',
    'BMXWAIST': 'Waist Circumference',
    'LBXHSCRP': 'HS C-Reactive Protein',
    'INDFMPIR': 'Poverty Ratio',
    'LBXGH': 'HbA1c'  # Renaming the response variable
})


m1_psql = tabulate(m1_df.head(15), headers='keys', tablefmt='psql')
print("\nFirst 15 obs of merged data for model 1 (m1_df)")
print(m1_psql)

# Check for the nan values in the merged_m1_df
has_nans = m1_df.isnull().any() # Check nans values for each column and return true if they have at least one nan value
has_nans = has_nans.any() # Return true if the whole data has atleast one nan value

# Display the rows with the nan values to inspect for any patterns
nan_rows = m1_df[m1_df.isnull().any(axis=1)]
print("\nFirst 20 rows which contain nan values in m1_df DataFrame")
print(tabulate(nan_rows.head(20), headers='keys', tablefmt='psql'))
print("\nPercentage of nan values in m1_df: {}".format(
    (nan_rows.shape[0]/m1_df.shape[0])*100
))

# Remove all the nan values
m1_df = m1_df.dropna()
print("\nCleaned data has nan values: {}".format(m1_df.isnull().any().any()))

# Generate statistics for continuous variables
continuous_vars = ['Age', 'BMI', 'Waist Circumference', 'BP', 'HS C-Reactive Protein', 'Poverty Ratio', 'Insulin Resistance Ratio', 'TC_to_HDL_Ratio', 'Non_HDL_Chol', 'HbA1c']
desc_stats = m1_df[continuous_vars].describe(percentiles=[.25, .5, .75])
desc_stats.loc['skew'] = m1_df[continuous_vars].skew()
desc_stats.loc['kurtosis'] = m1_df[continuous_vars].kurtosis()
print("\nDescriptive statistics of continuous variables")
print(tabulate(desc_stats, headers='keys', tablefmt='psql'))

# Generate freq table for categorical variables
def print_factor_table(data_frame, column):
    freq = data_frame[column].value_counts(dropna=False).reset_index()
    freq['Percentage'] = (freq['count'] / len(data_frame)) * 100
    print("\nFrequency table for variable {}: ".format(column))
    print(tabulate(freq, headers='keys', tablefmt='psql'))

print_factor_table(m1_df, 'Gender')
print_factor_table(m1_df, 'Race')

# Generate histograms for continuous variables
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(continuous_vars):
    sns.histplot(data=m1_df, x=var, ax=axes[i], kde=True, bins=30)
    axes[i].set_title(f"Distribution of {var}", fontweight='bold')
    axes[i].set_xlabel(f"{var}")
    axes[i].set_ylabel("Count")

# plt.delaxes(axes[13])
plt.tight_layout()
plt.show()

# Generate box plots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18, 8))
axes = axes.flatten()

for i, var in enumerate(continuous_vars):
    sns.boxplot(data=m1_df, x=var, ax=axes[i])
    axes[i].set_title(f"Box Plot of {var}", fontweight='bold')

# plt.delaxes(axes[13])
plt.tight_layout()
plt.show()

# Generate count plots for categorical variables
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

for i, var in enumerate(['Gender', 'Race']):
    sns.countplot(data=m1_df, x=var, ax=axes[i])
    axes[i].set_title(f"{var} Distribution", fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Generate pair plot of continuous variables and kernel density estimates (kde) on the diagonal
sns.pairplot(
    data=m1_df[continuous_vars], diag_kind='kde', corner=True
)
plt.suptitle("Variable Relationships")
plt.show()

# Generate heatmap for continuous variables for correlation analysis
fig = plt.figure(figsize=(24, 16))
corr_matrix = m1_df[continuous_vars].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
plt.title("Variable Correlations Heatmap", fontweight='bold')
plt.show()

# Find the best lambda value using BoxCox object and transform BMI
bmi_data = m1_df['BMI'].values
insulin_data = m1_df['Insulin Resistance Ratio'].values
bp_data = m1_df['BP'].values
inflammation_data = m1_df['HS C-Reactive Protein'].values
non_hdl_data = m1_df['Non_HDL_Chol'].values
chol_ratio_data = m1_df['TC_to_HDL_Ratio'].values
poverty_data = m1_df['Poverty Ratio']
HbA1c_data = m1_df['HbA1c']

transformer = BoxCox.__new__(BoxCox)

def transformer_func(object: object, data, lambda_range, lambda_step, var_names):
    object.data = data
    object.lambda_range = lambda_range
    object.lambda_step = lambda_step
    best_lambda, max_log_l, lambdas, log_ls = object.find_best_lambda()

    print(f"The best found lambda to normalize x values is: {best_lambda}")
    print(f"Maximum log likelihood is: {max_log_l}")

    transformer.current_lambda = best_lambda
    m1_df[var_names[1]] = object.box_cox_transform()

    # Visualize the transformation
    # Plot the maximum loglikelihood graph of box cox method
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, log_ls)
    plt.title('Box Cox Log Likelihood vs Lambdas')
    plt.xlabel('Lambdas')
    plt.ylabel('Log-likelihood')
    plt.axvline(best_lambda, color='r', linestyle='-.', label=f"Optimal Lambda ≈ {best_lambda:.3f}")
    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

    for i, var in enumerate(var_names):
        sns.histplot(data=m1_df, x=var, ax=axes[i], kde=True, bins=20)
        if 'Box_Cox' in f'{var}':
            axes[i].set_title(f"{var} transformed")

        elif not 'Box_Cox' in f'{var}':
            axes[i].set_title(f"{var} untransformed")
        axes[i].set_xlabel(f"{var}")
        axes[i].set_ylabel("Count")
    plt.show()

    return best_lambda, max_log_l, lambdas, log_ls

# Find the best lambda value using BoxCox object and transform bmi
transformer_func(object=transformer, data=bmi_data, lambda_range=[-2, 2], lambda_step=100, var_names=['BMI', 'BMI_Box_Cox'])

# Find the best lambda value using BoxCox object and transform Insulin Resistance Ratio
transformer_func(object=transformer, data=insulin_data, lambda_range=[-2, 2], lambda_step=100, var_names=['Insulin Resistance Ratio', 'Insulin_Resistance_Ratio_Box_Cox'])

# Find the best lambda value using BoxCox object and transform BP
transformer_func(object=transformer, data=bp_data, lambda_range=[-3, 2], lambda_step=100, var_names=['BP', 'BP_Box_Cox'])

# Find the best lambda value using BoxCox object and transform HS C-Reactive Protein
transformer_func(object=transformer, data=inflammation_data, lambda_range=[-2, 2], lambda_step=100, var_names=['HS C-Reactive Protein', 'HS_C_Box_Cox'])

# Find the best lambda value using BoxCox object and transform Non_HDL_Chol
transformer_func(object=transformer, data=non_hdl_data, lambda_range=[-2, 2], lambda_step=100, var_names=['Non_HDL_Chol', 'Non_HDL_Chol_Box_Cox'])

# Find the best lambda value using BoxCox object and transform TC_to_HDL_Ratio
transformer_func(object=transformer, data=chol_ratio_data, lambda_range=[-2, 2], lambda_step=100, var_names=['TC_to_HDL_Ratio', 'TC_to_HDL_Box_Cox'])


# Find the best lambda value using BoxCox object and transform Poverty Ratio
transformer_func(object=transformer, data=poverty_data, lambda_range=[-2, 2], lambda_step=100, var_names=['Poverty Ratio', 'Poverty_Ratio_Box_Cox'])

# Find the best lambda value using BoxCox object and transform HbA1c
transformer_func(object=transformer, data=HbA1c_data, lambda_range=[-4, -1], lambda_step=100, var_names=['HbA1c', 'HbA1c_Box_Cox'])


print(tabulate(m1_df.head(30), headers='keys', tablefmt='psql'))