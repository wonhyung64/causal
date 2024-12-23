#%%
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from numpy.random.mtrand import RandomState


class MF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)

        return out, user_embed, item_embed


def print_result(result_list, top_k_list):
    for k, top_k in enumerate(top_k_list):
        print(f"Top {top_k}")
        print(f"Avg. : {np.mean(np.array(result_list)[:, k]).round(6)}, Std. : {np.std(np.array(result_list)[:, k]).round(6)}")
        print()


#%% options
rng = RandomState(seed=None)
lr = 0.01
embedding_k = 200
num_epochs = 500
top_k_list = [10, 100]
batch_size = 4096


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"


# %% Category / Original
data_dir = "/Users/wonhyung64/Github/causal/UnbiasedLearningCausal/data/preprocessed/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf2.00_nr210"

train_df = pd.read_csv(f"{data_dir}/data_train.csv")
test_df = pd.read_csv(f"{data_dir}/data_test.csv")
valid_df = pd.read_csv(f"{data_dir}/data_vali.csv")

num_items = test_df["idx_item"].max() + 1
num_users = test_df["idx_user"].max() + 1
top_k_list.append(num_items)

df_train = train_df.sample(frac=1)
t_train = df_train.loc[:, "treated"].values
x_train = df_train.loc[:, ["idx_user", "idx_item"]].values
y_train = df_train.loc[:, 'outcome'].values


#%% outcome T=1 modeling
xt1_train = x_train[t_train==1]
yt1_train = y_train[t_train==1]

num_sample = len(xt1_train)
total_batch = num_sample // batch_size

model_t1 = MF(num_users, num_items, embedding_k)
model_t1 = model_t1.to(device)
optimizer = torch.optim.Adam(model_t1.parameters(), lr=lr)
loss_fcn = torch.nn.BCELoss()

all_items = np.arange(num_items)

for epoch in tqdm(range(1, num_epochs+1)):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    model_t1.train()

    for idx in range(total_batch):

        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = xt1_train[selected_idx]
        sub_x = torch.LongTensor(sub_x).to(device)
        sub_y = yt1_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        pred, user_embed, item_embed = model_t1(sub_x)

        total_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


#%% outcome T=0 modeling
xt0_train = x_train[t_train==0]
yt0_train = y_train[t_train==0]

num_sample = len(xt0_train)
total_batch = num_sample // batch_size

model_t0 = MF(num_users, num_items, embedding_k)
model_t0 = model_t0.to(device)
optimizer = torch.optim.Adam(model_t0.parameters(), lr=lr)
loss_fcn = torch.nn.BCELoss()

all_items = np.arange(num_items)

for epoch in tqdm(range(1, num_epochs+1)):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    model_t0.train()

    for idx in range(total_batch):

        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = xt0_train[selected_idx]
        sub_x = torch.LongTensor(sub_x).to(device)
        sub_y = yt0_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        pred, user_embed, item_embed = model_t0(sub_x)

        total_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


#%%
test_ate = test_df.groupby(["idx_user", "idx_item"])["causal_effect"].mean().reset_index()
users = test_ate["idx_user"].values
items = test_ate["idx_item"].values
true = test_ate["causal_effect"].values
x_test = test_ate[["idx_user", "idx_item"]].values
x_test = torch.LongTensor(x_test).to(device)

model_t1.eval()
pred_t1, user_embed, item_embed = model_t1(x_test)
pred_t1 = nn.Sigmoid()(pred_t1).detach().cpu().numpy()

model_t0.eval()
pred_t0, user_embed, item_embed = model_t0(x_test)
pred_t0 = nn.Sigmoid()(pred_t0).detach().cpu().numpy()

pred = (pred_t1 - pred_t0)

user_dcg_list, user_precision_list, user_ar_list = [], [], []
for u in tqdm(range(num_users)):

    user_idx = users==u
    user_pred = (pred[user_idx]).squeeze(-1)
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
