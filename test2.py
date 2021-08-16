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
import json as js
import os
import shutil as sh
import random
import itertools as ittl
from scipy.sparse import csr_matrix
import best_subset_funcs as fncs

def mkdif(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
calculate = True

print("starting")
iso_fs = ["Chimera_tail/iso1.csv","cis-trans/trans_iso1.csv", "cis-trans/11cis_iso1.csv", "cis-trans/9cis_iso1.csv"]
emb_fs = ["Chimera_tail/emb1.csv", "cis-trans/trans_emb1.csv", "cis-trans/11cis_emb1.csv", "cis-trans/9cis_emb1.csv"]
os.chdir("/home/nico/WORK/HUJI")
for iso_f, emb_f in zip(iso_fs, emb_fs):
    isos = pd.read_csv(iso_f,index_col=0)[["ex_en", "osc"]].values
    embs = pd.read_csv(emb_f,index_col=0)[["ex_en", "osc"]].values
    fol, fniso = os.path.split(iso_f)
    sbstfol = os.path.join(fol, "subset")
    mkdif(sbstfol)
    system =  "chimera" if fol == "Chimera_tail" else fniso[:-9] 
    if calculate:
        nbins = 10
        for factor in [3,4,5]:
            t1 = 0.0001
            thresh = 0.001
            itr = 3
            for arr, arr_name in zip([isos, embs], ["iso", "emb"]):
                deltaN = np.zeros([itr,2])
                deltaS = np.zeros([itr,2])
                sbst_lst, ws_lst = [], []
                print(arr_name, system, factor)
                for n in range(itr):
                    print(n)
                    sbst, ws = fncs.neighs_subset2D(arr, nbins, factor, t1, thresh)
                    sbst_lst.append(sbst)
                    ws_lst.append(ws)
                    deltaN[n] = np.array([np.average(arr[:,0], weights=arr[:,1]) - np.average(arr[:,0][sbst], weights=arr[:,1][sbst]*ws),
                         np.average(arr[:,1]) - np.average(arr[:,1][sbst], weights=ws)])
                idx = np.where(abs(deltaN).mean(axis=1) == abs(deltaN).mean(axis=1).min())[0][0]
                best_dN = deltaN[idx]
                sbst = sbst_lst[idx]
                ws =ws_lst[idx]
                jsfp = os.path.join(sbstfol, "{}_{}_todel_{}.json".format(system, arr_name, factor))
                with open(jsfp, "w") as  f:
                    print(system, arr_name, factor, ws.sum())
                    js.dump([sbst.tolist(), ws.tolist()], f)
    else:
        fol, fname = os.path.split(iso_f)
        for factor in [3,4,5]:
            for arr, arr_name in zip([isos, embs], ["iso", "emb"]):
                jsfp = os.path.join(sbstfol, "{}_{}_todel_{}.json".format(system, arr_name, factor))
                with open(jsfp, "r") as  f:
                    sbst, ws = np.array(js.load(f))
                sbst = sbst.astype("int")
                print(system, factor, ws.sum())
#                plotbins = 10
#                fig = plt.figure(figsize=(20, 10), dpi=150)
#                ax = fig.add_subplot(221)
#                _, bins, _t = ax.hist(arr[:,0], bins=plotbins, alpha=1, density=True,
#                                      color="red", histtype="step", linewidth=5, label="full")
#                ax.hist(arr[:,0][sbst], weights=ws, bins=bins, alpha=0.5, density=True,
#                        color="green", histtype="step", linewidth=5, linestyle="-", label="subset")
#                ax.set_xlabel(r"$\varepsilon$")
#                ax.set_ylabel("occurrence")
#                ax.legend()
#                ax.set_title(r"$\Delta <\varepsilon^{{{lbl}}}>$ = {d1:.5f}   $\Delta <f^{{{lbl}}}>$ = {d2:.5f}".format(**{
#                        "d1": np.average(arr[:,0], weights=arr[:,1]) - np.average(arr[:,0][sbst], weights=arr[:,1][sbst]*ws),
#                        "d2": np.average(arr[:,1]) - np.average(arr[:,1][sbst],  weights=ws),
#                        "lbl": arr_name}))
#                ax2 = fig.add_subplot(222)
#                _, bins, _t = ax2.hist(arr[:,1], bins=plotbins, alpha=1, density=True,
#                                       color="red", histtype="step", linewidth=5, label="full")
#                ax2.hist(arr[:,1][sbst], weights=ws, bins=bins, alpha=0.5, density=True,
#                         color="green", histtype="step", linewidth=5, linestyle="-", label="subset")
#                ax2.set_xlabel("$f$")
#                ax2.set_ylabel("occurrence")
#                ax2.legend()
#                other_arr = embs if arr_name == "iso" else isos
#                other_name = "emb" if arr_name == "iso" else "iso"
#                ax2.set_title(r"$\Delta <\varepsilon^{{{lbl}}}>$ = {d1:.5f}   $\Delta <f^{{{lbl}}}>$ = {d2:.5f}".format(**{
#                        "d1": np.average(other_arr[:,0], weights=other_arr[:,1]) - np.average(other_arr[:,0][sbst], weights=other_arr[:,1][sbst]*ws),
#                        "d2": np.average(other_arr[:,1]) - np.average(other_arr[:,1][sbst], weights=ws),
#                        "lbl": other_name}))
#                ax3 = fig.add_subplot(223)
#                Z3, xedges3, yedges3 = np.histogram2d(*arr.T, bins=10, density=True)
##                h3 = ax3.hist2d(*arr.T, bins=10, alpha=0.5, density=True)
#                h3 = ax3.pcolormesh(xedges3, yedges3, Z3)
#                ax3.set_xlabel(r"$\varepsilon$")
#                cb = fig.colorbar(h3, ax=ax3)
##                cb = fig.colorbar(h3[3], ax=ax3)
#                density = cb.get_ticks()
#                ax3.set_ylabel(r"$f$")
#                ax3.set_title("full")
#                ax4 = fig.add_subplot(224)
##                h4 = ax4.hist2d(*arr[sbst].T, weights=ws, bins=10, alpha=0.5, density=True)
#                Z4, xedges4, yedges4 = np.histogram2d(*arr[sbst].T, weights=ws, bins=10, density=True)
#                h4 = ax4.pcolormesh(xedges4, yedges4, Z4, vmin=density.min(), vmax=density.max())
#                ax4.set_xlabel(r"$\varepsilon$")
#                fig.colorbar(h4, ax=ax4)
##                fig.colorbar(h4[3], ax=ax4)
#                ax4.set_ylabel(r"$f$")
#                ax4.set_title("subset")
#                fig.suptitle("{}, factor = {}".format(arr_name, factor))
#                fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.075, wspace=0.25, hspace=0.3)
#                fig.savefig(jsfp[:-4]+"png")

#fig2 = plt.figure(figsize=(20, 10), dpi=150)
#ax = fig2.add_subplot(111)
#ax.plot(*arr.T, markerfacecolor="r", markersize=5, markeredgewidth=1, markeredgecolor="k", linestyle="", marker="o", alpha=0.35)  
#ax.plot(*arr[third].T, markerfacecolor="b", markersize=5, markeredgewidth=1, markeredgecolor="k", linestyle="", marker="o", alpha=0.5)  
#ax.bar(*exosc_ensemble(*arr.T), color="r", width=0.01, edgecolor="k", linewidth=2, alpha=0.65, zorder=50)  # ensemble iso
#ax.bar(*exosc_ensemble(*arr[third].T), color="b", width=0.01, edgecolor="k", linewidth=2, alpha=0.65, zorder=50)  # ensemble emb
##ax.hist(arr[third], bins=40, alpha=0.5, density=True, color="green", histtype="step", linewidth=5)
##ax.hist(arr[third], bins=20, alpha=0.5, density=True, color="green", histtype="step", linewidth=5, linestyle="dotted")
##ax.hist(arr[fifth], bins=bins, alpha=0.5, density=True, color="blue", histtype="step", linewidth=5)
##ax.hist(arr[tenth], bins=bins, alpha=0.3, density=True, color="orange", histtype="step", linewidth=5)
##        
#
###a,=ax.plot(*arr.T, linestyle="", marker="o", markeredgecolor="black", markerfacecolor="red", alpha=1, markersize=5)
###b,=ax.plot(*kept.T, linestyle="", marker="o", markeredgecolor="black", markerfacecolor="blue", alpha=0.5, markersize=10)
##
###ax.legend([a,b],["arr","subset"],loc="best",edgecolor="black")


