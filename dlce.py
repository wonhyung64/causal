#%%
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random.mtrand import RandomState


def func_sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return np.exp(x) / (1.0 + np.exp(x))


def print_result(result_list, top_k_list):
    for k, top_k in enumerate(top_k_list):
        print(f"Top {top_k}")
        print(f"Avg. : {np.mean(np.array(result_list)[:, k]).round(6)}, Std. : {np.std(np.array(result_list)[:, k]).round(6)}")
        print()


#%% options
rng = RandomState(seed=None)
capping_T = 0.1
capping_C = 0.1
with_IPS = True
lr = 0.003
embedding_k = 200
coeff_T = 1.
coeff_C = 1.
num_epochs = 500
top_k_list = [10, 100]

# %% Category / Original
data_dir = "/Users/wonhyung64/Github/causal/UnbiasedLearningCausal/data/preprocessed/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf2.00_nr210"

train_df = pd.read_csv(f"{data_dir}/data_train.csv")
test_df = pd.read_csv(f"{data_dir}/data_test.csv")
valid_df = pd.read_csv(f"{data_dir}/data_vali.csv")

num_items = test_df["idx_item"].max() + 1
num_users = test_df["idx_user"].max() + 1
top_k_list.append(num_items)

df_train = train_df.loc[train_df.loc[:, "outcome"] > 0, :]

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

df_train = df_train.sample(frac=1)
users = df_train.loc[:, "idx_user"].values
items = df_train.loc[:, "idx_item"].values
ITE = df_train.loc[:, 'ITE'].values

for epoch in tqdm(range(num_epochs)):
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

#%%
test_ate = test_df.groupby(["idx_user", "idx_item"])["causal_effect"].mean().reset_index()
users = test_ate["idx_user"].values
items = test_ate["idx_item"].values
true = test_ate["causal_effect"].values
pred = np.zeros(len(test_ate))

for n in np.arange(len(test_ate)):
    pred[n] = np.inner(user_factors[users[n], :], item_factors[items[n], :])

user_dcg_list, user_precision_list, user_ar_list = [], [], []
for u in tqdm(range(num_users)):

    user_idx = users==u
    user_pred = (pred[user_idx])
    user_pred = (user_pred - user_pred.min()) / (user_pred.max() - user_pred.min())
    user_true = true[user_idx]

    dcg_k_list, precision_k_list, ar_k_list = [], [], []
    for top_k in top_k_list:

        """ndcg@k"""
        log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
        pred_top_k_rel = user_true[np.argsort(-user_pred)][:top_k]
        dcg_k = (pred_top_k_rel / log2_iplus1).sum()
        dcg_k_list.append(dcg_k)

        """precision@k"""
        p_k = pred_top_k_rel.sum()
        precision_k_list.append(p_k)

        """average rank@k"""
        ar_k = np.sum(-(np.arange(1,top_k+1)) * pred_top_k_rel)
        ar_k_list.append(ar_k)

    user_dcg_list.append(dcg_k_list)
    user_precision_list.append(precision_k_list)
    user_ar_list.append(ar_k_list)

print("CDCG")
print_result(user_dcg_list, top_k_list)

print("CP")
print_result(user_precision_list, top_k_list)

print("CAR")
print_result(user_ar_list, top_k_list)


# %%
