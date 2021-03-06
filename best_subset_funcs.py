#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:03:31 2021

@author: nico
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:09:51 2021

@author: nico
"""
import  numpy as np
#from timeit import timeit
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import random
import itertools as ittl
from scipy.sparse import csr_matrix

distN = lambda x,y: np.linalg.norm(x-y)
dist1 = lambda x,y: abs(x-y)
#arr_in_list = lambda a, l: np.any(np.all(a == l, axis=1))

def exosc_ensemble(ex, osc):
    return (np.average(ex,weights=osc), osc.sum()/osc.shape[0])

def intersect_lists(ll):  # tested to be faster than other possible methods
    intersec = set(ll[0])
    for l in ll[1:]:
        intersec = intersec & set(l)
    return list(intersec)

def nearby_points_dict(arr, thresh):
    dist = distN if len(arr.shape) == 2 else dist1
    d = {}
    for n1,p1 in enumerate(arr):
        mx, mn = p1 + thresh, p1 - thresh
        square = np.where(np.logical_and((arr < mx).all(axis=1), (arr > mn).all(axis=1)))[0]
        circle = [n2 for n2 in square if dist(p1, arr[n2]) < thresh and n2 != n1]
        d[n1] = circle
    return d
       
def nearby_points(arr, thresh):
    dist = distN if len(arr.shape) == 2 else dist1
    l = []
    for n1,p1 in enumerate(arr):
        mx, mn = p1 + thresh, p1 - thresh
        square = np.where(np.logical_and((arr < mx).all(axis=1), (arr > mn).all(axis=1)))[0]
        circle = [n2 for n2 in square if dist(p1, arr[n2]) < thresh and n2 != n1]
        if circle: 
            l.append(n1)
    return l    

def nearby_points2(arr, thresh):
    dist = distN if len(arr.shape) == 2 else dist1
    l = []
    for n1,p1 in enumerate(arr):
        mx, mn = p1 + thresh, p1 - thresh
        square = np.where(np.logical_and((arr < mx).all(axis=1), (arr > mn).all(axis=1)))[0]
        l.extend([n2 for n2 in square if dist(p1, arr[n2]) < thresh and n2 not in l and n2 != n1])
    return l    

def coalesce(arr, thresh):
    if type(arr) == list:
        arr = np.array(arr)
    dist = distN if len(arr.shape) == 2 else dist1
    l = []
    weights = np.ones(len(arr))
    for n1, p1 in enumerate(arr):
        if n1 in l:
            continue
        mx, mn = p1 + thresh, p1 - thresh
        square = np.where(np.logical_and((arr < mx).all(axis=1), (arr > mn).all(axis=1)))[0]
        to_ext = [n2 for n2 in square if dist(p1, arr[n2]) < thresh and n2 not in l and n2 != n1]
        weights[n1] += sum([weights[i] for i in to_ext])
        l.extend(to_ext)
    points = np.delete(np.arange(len(arr)),  l)
    weights = weights[[i for i in range(len(arr)) if i not in l]]
    return points, weights

def reduce_to(arr, des_size, t0, t_inc):
    r = arr
    t = t0
    while r.shape[0] > des_size:
        tmp, w = coalesce(arr, t)
        r = arr[tmp]
        t += t_inc
    return tmp, w

def reduce_bisect(arr, des_size, t1, thresh):  # works with N dimensions
    t2 = distN(arr.max(axis=0),arr.min(axis=0))
    assert arr[coalesce(arr, t1)[0]].shape[0] > des_size
    it = 0
    while t2 -t1 > thresh:
        it += 1
        t = 0.5*(t2 + t1)
        s = arr[coalesce(arr, t)[0]].shape[0]
        if s == des_size:
            break
        if s > des_size:
            t1 = t
        else:
            t2 = t
    return coalesce(arr, 0.5*(t2 + t1))

def get_bins1D(arr, bins=5, range_=()):  
    """
    """
    intbins = True if type(bins) == int else False
    index = np.arange(arr.shape[0])
    shape = [bins, len(index)] if intbins else [len(bins), len(index)] # shape can be inferred but it is probably faster to give it
    if not range_:
        range_ = (arr.min(), arr.max())
    else:
        in_range = np.logical_and(arr >= range_[0], arr <= range_[1])
        index = index[in_range]  # frame numbers
        arr = arr[in_range]
    digitized = (float(bins)/(range_[1] - range_[0])*(arr - range_[0])).astype(int)\
    if intbins else np.digitize(arr, bins) - 1   # array of what bin each frame is in
    if intbins and bins in digitized:
        digitized[digitized == bins] = bins -1  # so that last bin includes max
    S = csr_matrix((arr, [digitized, index]), shape=shape)
    to_return = np.split(S.indices, S.indptr[1:-1])
    return to_return

def get_bins2D(arrN2, bins=5, range_=[], flatten=False):  
    """
    """
    x, y = arrN2.T
    if type(bins) == int:
        bins = [bins, bins]
    elif type(bins) == np.ndarray:
        bins = [bins, bins]
    intbins = [type(b) == int for b in bins]
    if range_ and type(range_) == list and type(range_[0]) != list:
        range_ = (range_, range_)
    if not range_:
        range_ = [[],[]]
    digitized = []
    idx = []
    for n,i in enumerate([x,y]):
        index = np.arange(i.shape[0])
        if not range_[n]:
            range_[n] = [i.min(), i.max()]
        else:
            in_range = np.logical_and(i >= range_[n][0], i <= range_[n][1])
            index = index[in_range]  # frame numbers
        idx.append(index)
    index = intersect_lists(idx)
    for n,i in enumerate([x,y]):
        digs = (float(bins[n])/(range_[n][1] - range_[n][0])*(i[index] - range_[n][0])).astype(int) \
        if intbins[n] else np.digitize(i[index], bins[n]) - 1# array of what bin each frame is in
        if intbins[n] and bins[n] in digs:
            digs[digs == bins[n]] = bins[n] -1  # so that last bin includes max
        digitized.append(digs)         
    digitized = np.stack(digitized, axis=1)
    flat_dig = [(bins[0] if intbins[0] else len(bins[0]))*i[0] + i[1] for i in digitized]
    shape = [(bins[0] if intbins[0] else len(bins[0]))*(bins[1] if intbins[1] else len(bins[1])), len(index)]
    S = csr_matrix((index, [flat_dig, index]), shape=shape)
    flat_bins = np.split(S.indices, S.indptr[1:-1])
    to_return = flat_bins if flatten else np.array(flat_bins, dtype="object").reshape(
        [b if intbins[n] else len(b) for n,b in enumerate(bins)])
    return to_return

def get_bins(arr, bins=5, range_=[], flatten=False):
    if len(arr.shape) == 1:
        return get_bins1D(arr, range_=range_, bins=bins)
    elif len(arr.shape) > 2:
        raise NotImplementedError("get_bins now only works with 1d- and 2d-arrays")
    elif arr.shape[1] > 2:
        raise NotImplementedError("get_bins now only works with 1 and 2 dimensions")
    elif arr.shape[1] == 1:
        return get_bins1D(arr.reshape(-1), range_=range_, bins=bins)
    elif arr.shape[1] == 2:    
        return get_bins2D(arr, bins=bins, range_=range_, flatten=flatten)
    
def reduce_bins(arr, bins, factor, t1, thresh):
    bins = get_bins(arr, bins=bins, flatten=True)
    nbins, weights = [], []
    for b in bins:
        if len(b):
            bidx, ws = reduce_bisect(arr[b], round(b.size/factor), t1, thresh)
            idx = b[bidx]
            nbins.append(idx)
            weights.append(ws)
        else:
            nbins.append(np.array([], dtype="int"))
            weights.append(np.array([], dtype="int"))
    return nbins, weights

def subset(arr, bins, factor, t1, thresh):
    abins = get_bins(arr, bins=bins)
    sbins, weights = reduce_bins(arr, bins, factor, t1, thresh)
    abc = np.array([i.size for i in abins])
    sbc = np.array([i.size for i in sbins])
    diff = abc - sbc*factor
    less, iless = diff[diff > 0], np.where(diff > 0)[0]
    more, imore = abs(diff[diff < 0]), np.where(diff < 0)[0]
    oless = np.argsort(less)[::-1]
    omore = np.argsort(more)[::-1]
    todel = {}
    toadd = {}
    for n in omore:
        if round(more[n]/factor) > 0:
            pool = sbins[imore[n]].tolist()
            topick = round(more[n]/factor)
            if len(pool) == topick:
                topick -= 1
            todel[imore[n]] = random.sample(pool, topick)  
    for n in oless:
        if round(less[n]/factor):
            toadd[iless[n]] = random.sample([i for i in abins[iless[n]] if i not in sbins[iless[n]]], round(less[n]/factor))  # here
    for k,v in todel.items():
        w_inc = sum([weights[k][n] for n,i in enumerate(sbins[k]) if i in v])  # tot weight increment
        weights[k] = np.array([weights[k][n] for n,i in enumerate(sbins[k]) if i not in v])
        weights[k] += w_inc/len(weights[k])
        sbins[k] = np.array([i for i in sbins[k] if i not in v])
    for k, v in toadd.items():
        ws = np.ones(len(v))
        weights[k] -= sum(ws)/len(weights[k])
        weights[k] = np.append(weights[k], ws)
        sbins[k] = np.append(sbins[k], v)
    return np.concatenate(sbins, axis=None).astype("int"), np.concatenate(weights, axis=None)
#

def subset2D(arr, bins, factor, t1, thresh, concat=True):
    abins = get_bins(arr, bins=bins, flatten=True)
    sbins, weights = reduce_bins(arr, bins, factor, t1, thresh)
    abc = np.array([i.size for i in abins])
    sbc = np.array([i.size for i in sbins])
    diff = abc - sbc*factor
    less, iless = diff[diff > 0], np.where(diff > 0)[0]
    more, imore = abs(diff[diff < 0]), np.where(diff < 0)[0]
    oless = np.argsort(less)[::-1]
    omore = np.argsort(more)[::-1]
    todel = {}
    toadd = {}
    for n in omore:
        if round(more[n]/factor) > 0:
            pool = sbins[imore[n]].tolist()
            topick = round(more[n]/factor)
            if len(pool) == topick:
                topick -= 1
            todel[imore[n]] = random.sample(pool, topick)  # here
    for n in oless:
        if round(less[n]/factor):
            toadd[iless[n]] = random.sample([i for i in abins[iless[n]] if i not in sbins[iless[n]]], round(less[n]/factor))  # here
    for k,v in todel.items():
        w_inc = sum([weights[k][n] for n,i in enumerate(sbins[k]) if i in v])  # tot weight increment
        weights[k] = np.array([weights[k][n] for n,i in enumerate(sbins[k]) if i not in v])
        weights[k] += w_inc/len(weights[k])
        sbins[k] = np.array([i for i in sbins[k] if i not in v])
    for k, v in toadd.items():
        ws = np.ones(len(v))
        weights[k] -= sum(ws)/len(weights[k])
        weights[k] = np.append(weights[k], ws)
        sbins[k] = np.append(sbins[k], v)
    to_return = (np.concatenate(sbins, axis=None).astype("int"), np.concatenate(weights, axis=None)) if concat else (sbins, weights)
    return to_return

didx2flat = lambda idx, y_shape: idx[0]*y_shape + idx[1]
fidx2dual = lambda idx, y_shape: (idx//y_shape, idx % y_shape)

def neighbours2_didx(didx):
       xs, ys = [[i - 1, i, i + 1] for i in didx]
       neighs = list(ittl.product(xs, ys))
       neighs.remove(didx)
       return neighs

def neighbours2_fidx(fidx, y_shape):
    didx = fidx2dual(fidx, y_shape)
    return [didx2flat(i, y_shape) for i in neighbours2_didx(didx)]

def neighbours2D(idx, *args):
    if args:
        y_shape = args[0]
    if type(idx) in [float, np.float32, np.float64]:
        idx = int(idx)
    if type(idx) in [int, np.int32, np.int64]:
        return neighbours2_fidx(idx, y_shape)
    else:
        return neighbours2_didx(idx)
    
neighbours1D = lambda idx: (idx - 1, idx + 1)

def neighbours(idx, *args, dim=2):
    return neighbours2D(idx, *args) if dim == 2 else neighbours1D(idx)

def neighs_subset2D(arr, bins, factor, t1, thresh):
    sbins, weights = subset2D(arr, bins, factor, t1, thresh, concat=False)
    sbc = np.array([len(i) for i in sbins])
    abins = get_bins(arr, bins=bins, flatten=True)
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
    assert weights_concat.sum() == float(len(arr))  # comment out later
    
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
            neighs = [i for i in neighbours2_fidx(s, bins) if i not in done and 0 <= i < imax] 
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
                todel.extend(sample)
                unsampled = [i for i in pool if i not in sample]
                deletions.append((unsampled, sample))
    toadd = []
    done = []
    additions = []
    for n in range(1, factor):
        sels = np.where(diff == n)[0]
        for s in sels:
            neighs = [i for i in neighbours2_fidx(s, bins) if i not in done and 0 <= i < imax] 
            if not neighs:
                continue
            done.extend(neighs)
            nsbc = sbc[neighs].sum()
            nabc = abc[neighs].sum() 
            d = nabc - nsbc*factor
            if round(d/factor) > 0:
                add_to = [i for i in np.concatenate([sbins_idx[n] for n in neighs]) if i not in todel]
                pool = [i for i in np.concatenate([abins[n] for n in neighs]) if i not in add_to]
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
    nweights = weights_concat[indexes]
    nweights = np.append(nweights, np.ones(len(toadd)))
    sbst = sbst.astype("int")
    return sbst, nweights