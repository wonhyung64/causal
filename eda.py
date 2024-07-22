#%%
import os
import numpy as np
import pandas as pd


rate_prior = 0.1

# %%
data_root = "./UnbiasedLearningCausal/data"
raw_dir = f"{data_root}/raw"
preprocess_dir = f"{data_root}/preprocessed"
product_dir = f"{preprocess_dir}/dunn_mailer_10_10_1_1"

#%%
promo_df = pd.read_csv(f"{raw_dir}/causal_data.csv")
promo_df.head(5)
promo_df["mailer"].unique()


#%%
trans_df = pd.read_csv(f"{raw_dir}/transaction_data.csv")
trans_df.head(5)

#%%
detail_df = pd.read_csv(f"{raw_dir}/product.csv")
detail_df["SUB_COMMODITY_DESC"].nunique()

#%%
product_df = pd.read_csv(f"{product_dir}/cnt_logs.csv")
product_df.head(5)


#%% 
product_df.loc[:, 'num_control'] = product_df.loc[:, 'num_visit'] - product_df.loc[:, 'num_treatment']
product_df.loc[:, 'num_control_outcome'] = product_df.loc[:, 'num_outcome'] - product_df.loc[:, 'num_treated_outcome']

#%% get item means
df_mean = product_df.loc[:, ["idx_item", 'num_treated_outcome', 'num_control_outcome',
                                'num_treatment', 'num_control', 'num_outcome', 'num_visit']]
df_mean = df_mean.groupby("idx_item", as_index=False).mean()
df_mean = df_mean.rename(columns={'num_treated_outcome': 'num_treated_outcome_mean',
                                    'num_control_outcome': 'num_control_outcome_mean',
                                    'num_treatment': 'num_treatment_mean',
                                    'num_control': 'num_control_mean',
                                    'num_outcome': 'num_outcome_mean',
                                    'num_visit': 'num_visit_mean'})

#%% merge
merged_df = pd.merge(product_df, df_mean, on=["idx_item"], how='left')

merged_df.loc[:, 'prob_outcome_treated'] = \
            (merged_df.loc[:, 'num_treated_outcome'] + rate_prior * merged_df.loc[:, 'num_treated_outcome_mean']) / \
            (merged_df.loc[:, 'num_treatment'] + rate_prior * merged_df.loc[:, 'num_treatment_mean'])

merged_df.loc[:, 'prob_outcome_control'] = \
    (merged_df.loc[:, 'num_control_outcome'] + rate_prior * merged_df.loc[:,'num_control_outcome_mean']) / \
    (merged_df.loc[:, 'num_control'] + rate_prior * merged_df.loc[:, 'num_control_mean'])

merged_df.loc[:, 'prob_outcome'] = \
    (merged_df.loc[:, 'num_outcome'] + rate_prior * merged_df.loc[:, 'num_outcome_mean']) / \
    (merged_df.loc[:, 'num_visit'] + rate_prior * merged_df.loc[:, 'num_visit_mean'])

#%%
num_data = merged_df.shape[0]
num_users = np.max(merged_df.loc[:, "idx_user"].values) + 1
num_items = np.max(merged_df.loc[:, "idx_item"].values) + 1

#%%
path = "/Users/wonhyung64/Github/causal/UnbiasedLearningCausal/data/preprocessed/dunn_mailer_10_10_1_1/original_rp0.90"
train_df = pd.read_csv(f"{path}/data_train.csv")
valid_df = pd.read_csv(f"{path}/data_vali.csv")
test_df = pd.read_csv(f"{path}/data_test.csv")

_df.__len__() / num_data
train_df["idx_time"].unique()
test_df["idx_time"].unique()
valid_df["idx_time"].unique()
train_df.sample(5)


merged_df
train_df[(train_df["idx_user"]==0) & (train_df["idx_item"]==10)]
merged_df[(merged_df["idx_user"]==20) & (merged_df["idx_item"]==5)]
train_df[(train_df["causal_effect"] == 0.) & (train_df["treated"] == 1) & (train_df["outcome"] == 1)]
train_df.loc[[37691045, 9350, 11969567, 47152297], :]

valid_df
test_df
merged_df[()]
merged_df["treated"] = 
def assign_treatment(self):
        self.df_data.loc[:, self.colname_treatment] = 0
        bool_treatment = self.df_data.loc[:, self.colname_propensity] > np.random.rand(self.num_data)
        self.df_data.loc[bool_treatment, self.colname_treatment] = 1

    def assign_outcome(self):
        self.df_data.loc[:, self.colname_outcome] = 0
        prob = np.random.rand(self.num_data)
        self.df_data.loc[:, self.colname_outcome_treated] = (self.df_data.loc[:, 'prob_outcome_treated'] >= prob) * 1.0
        prob = np.random.rand(self.num_data)
        self.df_data.loc[:, self.colname_outcome_control] = (self.df_data.loc[:, 'prob_outcome_control'] >= prob) * 1.0

        self.df_data.loc[:, self.colname_outcome] = \
            self.df_data.loc[:, self.colname_treatment] * self.df_data.loc[:, self.colname_outcome_treated] + \
            (1 - self.df_data.loc[:, self.colname_treatment]) * self.df_data.loc[:, self.colname_outcome_control]

    def assign_effect(self):
        self.df_data.loc[:, self.colname_effect] = \
            self.df_data.loc[:, self.colname_outcome_treated] - self.df_data.loc[:,self.colname_outcome_control]
        self.df_data.loc[:, self.colname_expectation] = \
            self.df_data.loc[:, 'prob_outcome_treated'] - self.df_data.loc[:, 'prob_outcome_control']

#%%
import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def indicator_function(x):
    return np.where(x < 0, 1, 0)

def log_exp_function(x, w):
    return np.log(1 + np.exp(-w * x))

# Define the range of x values
x_values = np.linspace(-10, 10, 400)

# Calculate y values for each function
y_indicator = indicator_function(x_values)


#%%
y_log_exp = log_exp_function(x_values, w=1) + 0.3
log_exp_function(-0.5,1)
# Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_indicator, label='1(x < 0)', color='blue', linestyle='--')
plt.plot(x_values, y_log_exp, label='log(1 + exp(-wx))', color='red')

# Add labels and legend
plt.title('Graphical Explanation of 1(x < 0) <= log(1 + exp(-wx))')
plt.xlabel('x')
plt.ylabel('Function value')
plt.legend()

# Show the plot
plt.grid(True)
plt.ylim(-0.5, 2)  # Set y-limits for better visualization
plt.show()