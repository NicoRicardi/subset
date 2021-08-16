#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:36:56 2021

@author: nico
"""

import  numpy as np
#from timeit import timeit
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import json as js
import os
import shutil as sh
import random
import itertools as ittl
from scipy.sparse import csr_matrix
import best_subset_funcs as fncs
import copy as cp

fname = "/home/nico/WORK/HUJI/cis-trans/9cis_iso1.csv"
arr = pd.read_csv(fname,index_col=0)[["ex_en", "osc"]].values

#t1 = 0.0001
#thresh = 0.001
t1 = 0.0001

thresh = 0.001
bins = 10
factor = 5
concat = False

sbins, weights = fncs.subset2D(arr, bins, factor, t1, thresh, concat=False)
sbc = np.array([len(i) for i in sbins])
abins = fncs.get_bins(arr, bins=bins, flatten=True)
abc = np.array([len(i) for i in abins])
diff = abc - sbc*factor

cnt = 0
split_points = []
for b in sbins[:-1]:
    cnt += len(b)
    split_points.append(cnt)
sbins_concat = np.concatenate(sbins, axis=None)
sbins_idx = np.split(np.arange(len(sbins_concat)), split_points)
weights_concat = np.concatenate(weights, axis=None)
assert weights_concat.sum() - float(len(arr)) < 1e-6 # comment out later
print("1", weights_concat.sum())

cnt = 0
split_points = []
for b in abins[:-1]:
    cnt += len(b)
    split_points.append(cnt)
abins_idx = np.split(np.arange(len(arr)), split_points)

todel = []
done = []
deletions = []
imax = len(sbins)
for n in range(-factor + 1, 0):
    sels = np.where(diff == n)[0]
    for s in sels:
        neighs = [i for i in fncs.neighbours2_fidx(s, bins) if i not in done and 0 <= i < imax] 
        if not neighs:
            continue
        done.extend(neighs)
        nsbc = sbc[neighs].sum()
        nabc = abc[neighs].sum() 
        if nabc == 1:
            continue
        d = nsbc*factor - nabc
        if round(d/factor) > 0:
            pool = np.concatenate([sbins_idx[n] for n in neighs]).tolist()
            topick = round(d/factor)
            if len(pool) == topick:
                topick -= 1
            sample = random.sample(pool, topick)
            todel.extend(sample)
            unsampled = [i for i in pool if i not in sample]
            deletions.append((unsampled, sample))
toadd = []
done = []
additions = []
for n in range(1, factor):
    sels = np.where(diff == n)[0]
    for s in sels:
        neighs = [i for i in fncs.neighbours2_fidx(s, bins) if i not in done and 0 <= i < imax] 
        if not neighs:
            continue
        done.extend(neighs)
        nsbc = sbc[neighs].sum()
        nabc = abc[neighs].sum() 
        d = nabc - nsbc*factor
        if round(d/factor) > 0:
            add_to = [i for i in np.concatenate([sbins_idx[n] for n in neighs]) if i not in todel]
            pool = [i for i in np.concatenate([abins[n] for n in neighs]) if i not in add_to] #asdf
            sample = random.sample(pool, round(d/factor))
            toadd.extend(sample)  
            additions.append((add_to, sample))

indexes = [i for i in range(len(sbins_concat)) if i not in todel]
sbst = sbins_concat.copy()[indexes]
sbst = np.append(sbst, toadd)
print("weight that will be removed", weights_concat.sum() - weights_concat[indexes].sum())
print("weight that will be added", len(toadd))
for del_ in deletions:
    print("will remove", weights_concat[del_[1]].sum())
#    print("adding to each unsampled", weights_concat[del_[1]].sum()/len(weights_concat[del_[0]]))
    weights_concat[del_[0]] += weights_concat[del_[1]].sum()/len(weights_concat[del_[0]])
print("2", weights_concat.sum())
for add in additions:
    print("will add", len(add[1]))
#    print("removing from each unsampled", len(add[1])/len(weights_concat[add[0]]))
    weights_concat[add[0]] -= len(add[1])/len(weights_concat[add[0]])
print("3", weights_concat.sum())
#nweights = weights_concat
nweights = weights_concat[indexes]
print("4", nweights.sum())
nweights = np.append(nweights, np.ones(len(toadd)))
sbst = sbst.astype("int")
print("sum is {:.5f}".format(nweights.sum()))
