#%%
import numpy as np
import pandas as pd



# %%
data_dir = "./data/dunnhumby"
product = pd.read_csv(f"{data_dir}/product.csv")
hh_demographic = pd.read_csv(f"{data_dir}/hh_demographic.csv")
coupon = pd.read_csv(f"{data_dir}/coupon.csv")
transaction = pd.read_csv(f"{data_dir}/transaction_data.csv")
transaction["WEEK_NO"].nunique()