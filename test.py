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

fname = "/home/nico/WORK/HUJI/cis-trans/trans_iso1.csv"
arr = pd.read_csv(fname,index_col=0)[["ex_en", "osc"]].values

t1 = 0.0001
thresh = 0.001
bins = 5
factor = 4
concat = False


sbins, weights = fncs.subset2D(arr, bins, factor, t1, thresh, concat=False)
sbc = np.array([len(i) for i in sbins])
abins = fncs.get_bins(arr, bins=bins, flatten=True)
abc = np.array([len(i) for i in abins])
diff = abc - sbc*factor

cnt = 0
split_points = []
for b in sbins:
    cnt += len(b)
    split_points.append(cnt)
sbins_concat = np.concatenate(sbins, axis=None)
sbins_idx = np.split(np.arange(len(sbins_concat)), split_points)
weights_concat = np.concatenate(weights, axis=None)

cnt = 0
split_points = []
for b in abins:
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
        d = nsbc*factor - nabc
        if round(d/factor) > 0:
            pool = np.concatenate([sbins_idx[n] for n in neighs]).tolist()
            topick = round(d/factor)
            if len(pool) == topick:
                topick -= 1
            sample = random.sample(pool, topick)
            if sample:
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
            add_to = np.concatenate([sbins_idx[n] for n in neighs])
            pool = [i for i in np.concatenate([abins_idx[n] for n in neighs]) if i not in add_to]
            sample = random.sample(pool, round(d/factor))
            toadd.extend(sample)  
            additions.append((add_to, sample))

indexes = [i for i in range(len(sbins_concat)) if i not in todel]
sbst = sbins_concat.copy()[indexes]
sbst = np.append(sbst, toadd)
for del_ in deletions:
    weights_concat[del_[0]] += weights_concat[del_[1]].sum()/len(weights_concat[del_[0]])
for add in additions:
    weights_concat[add[0]] -= len(add[1])/len(weights_concat[add[0]])
#nweights = weights_concat[indexes]
#nweights = np.append(nweights, np.ones(len(toadd)))
#sbst = sbst.astype("int")
#
##sbst, weights = fncs.neighs_subset2D(arr, bins, factor, t1, thresh)
##abins = fncs.get_bins(arr, bins=bins, flatten=True)
#range_ = [[arr[:,0].min(),arr[:,0].max()], [arr[:,1].min(), arr[:,1].max()]]
#asbst = arr[sbst]
#sbins2 = fncs.get_bins2D(asbst, bins=bins, range_=range_, flatten=True)
#sum_ = 0
#for n in range(len(sbins2)):
#    sb, ab = sbins2[n], abins[n]
#    if sb.size > 0 or ab.size > 0:
#        sum_ += nweights[sb].sum()
##        print("array", ab.size,"sum weights", ws[sb].sum())
#        if abs(ab.size - nweights[sb].sum()) > 0.01:
#            print(n, "array", ab.size,"sum weights", nweights[sb].sum())
