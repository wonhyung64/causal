#%%
import random
import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState


def func_sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return np.exp(x) / (1.0 + np.exp(x))


#%% options
rng = RandomState(seed=None)
capping_T = 0.1
capping_C = 0.1
with_IPS = True
metric = "AR_logi"
lr = 0.003
embedding_k = 200
with_bias = False
coeff_T = 1.
coeff_C = 1.

# %% Category / Original
data_dir = "/Users/wonhyung64/Github/causal/UnbiasedLearningCausal/data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40"

raw_df = pd.read_csv("/Users/wonhyung64/Github/causal/UnbiasedLearningCausal/data/preprocessed/dunn_cat_mailer_10_10_1_1/cnt_logs.csv")

train_df = pd.read_csv(f"{data_dir}/data_train.csv")
test_df = pd.read_csv(f"{data_dir}/data_test.csv")
valid_df = pd.read_csv(f"{data_dir}/data_vali.csv")

num_items = test_df["idx_item"].max() + 1
num_users = test_df["idx_user"].max() + 1

df_train = train_df.loc[train_df.loc[:, "outcome"] > 0, :] # need only positive outcomes

bool_cap = np.logical_and(df_train.loc[:, "propensity"] < capping_T, df_train.loc[:, "treated"] == 1)
if np.sum(bool_cap) > 0:
    df_train.loc[bool_cap, "propensity"] = capping_T

bool_cap = np.logical_and(df_train.loc[:, "propensity"] > 1 - capping_C, df_train.loc[:, "treated"] == 0)
if np.sum(bool_cap) > 0:
    df_train.loc[bool_cap, "propensity"] = 1 - capping_C

if with_IPS: # point estimate of individual treatment effect (ITE) <- for binary outcome abs(ITE) = IPS
    df_train.loc[:, 'ITE'] =  df_train.loc[:, "treated"] * df_train.loc[:, "outcome"]/df_train.loc[:, "propensity"] - \
                                (1 - df_train.loc[:, "treated"]) * df_train.loc[:, "outcome"]/(1 - df_train.loc[:, "propensity"])
else:
    df_train.loc[:, 'ITE'] =  df_train.loc[:, "treated"] * df_train.loc[:, "outcome"]  - \
                                (1 - df_train.loc[:, "treated"]) * df_train.loc[:, "outcome"]


#%% Train
user_factors = rng.normal(loc=0, scale=0.1, size=(num_users, embedding_k))
item_factors = rng.normal(loc=0, scale=0.1, size=(num_items, embedding_k))

while True:
    df_train = df_train.sample(frac=1)
    users = df_train.loc[:, "idx_user"].values
    items = df_train.loc[:, "idx_item"].values
    ITE = df_train.loc[:, 'ITE'].values

    for n in np.arange(len(df_train)):

        u = users[n]
        i = items[n]

        while True:
            j = random.randrange(num_items)
            if i != j:
                break

        u_factor = user_factors[u, :]
        i_factor = item_factors[i, :]
        j_factor = item_factors[j, :]

        diff_rating = np.sum(u_factor * (i_factor - j_factor))

        if ITE[n] >= 0:
            coeff = ITE[n] * coeff_T * func_sigmoid(-coeff_T * diff_rating) # Z=1, Y=1
        else:
            coeff = ITE[n] * coeff_C * func_sigmoid(coeff_C * diff_rating) # Z=0, Y=1

        user_factors[u, :] += \
            lr * (coeff * (i_factor - j_factor))
        item_factors[i, :] += \
            lr * (coeff * u_factor)
        item_factors[j, :] += \
            lr * (-coeff * u_factor)


    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])

        # pred = 1 / (1 + np.exp(-pred))
        return pred
