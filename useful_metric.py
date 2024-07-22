#%%
import numpy as np


def average_rank(rank: np.array) -> np.array:
    return -rank

def utility_fcn(gt_effect, rank, weight_fcn):
    return np.mean(gt_effect * weight_fcn(rank))
    

# %% Case 1
effect1 = np.array([1, -1, 1, 0, 0, 0,])
effect2 = np.array([0, 0, 0, 1, -1, 1,])

rank1 = np.array([1,6,2,3,4,5])
rank2 = np.array([1,5,3,2,6,4])


np.mean([
    utility_fcn(effect1, rank1, average_rank),
    utility_fcn(effect2, rank1, average_rank),
])

np.mean([
    utility_fcn(effect1, rank2, average_rank),
    utility_fcn(effect2, rank2, average_rank),
])


# %% Case 2
effect1 = np.array([0, 1, -1, 1, 0, 0,])
effect2 = np.array([ 0, 0, 1, -1, 1,0,])

rank1 = np.array([3,1,5,6,2,4])
rank2 = np.array([5,2,3,4,1,6])
rank3 = np.array([3,2,6,1,5,4])


np.mean([
    utility_fcn(effect1, rank1, average_rank),
    utility_fcn(effect2, rank1, average_rank),
])

np.mean([
    utility_fcn(effect1, rank2, average_rank),
    utility_fcn(effect2, rank2, average_rank),
])

np.mean([
    utility_fcn(effect1, rank3, average_rank),
    utility_fcn(effect2, rank3, average_rank),
])


# %%
