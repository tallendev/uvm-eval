#!/usr/bin/python3
#import experimental_models
#from experimental_models import opts, metrics
import sys
import argparse
import sys
import os
import argparse
from os.path import basename, splitext, dirname
import shutil

from natsort import humansorted

from glob import glob

import numpy as np

import itertools

import Batch
from Batch import Experiment

import subprocess

from pathlib import Path

import itertools, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from statistics import mean
from itertools import groupby


marker="."
markersize=1

#apps = ["cublas-long"]
apps = ["cublas", "linear", "random", "stream", "hpgmg", "cufft", "cublas-long", "hpgmg-parallel", "gauss-seidel"]

#apps = ["cublas", "cublas-long", "linear", "random", "stream", "hpgmg", "tealeaf", "cufft", "tealeaf-parallel", "hpgmg-parallel", "cublas-long", "gauss_seidel"]
#apps = ["gauss-seidel", "cufft"]
#apps = ["cublas"]
#apps = ["hpgmg"]
#apps = ["cublas", "hpgmg"]
#apps = ["cublas", "linear", "random", "stream", "hpgmg", "cufft"]

sizes = {"cublas" : [16384, 32768], "linear" : [1048576, 8388608], "random" : [1048576, 8388608], "stream": [250001408, 750004224], "hpgmg": ["77", "88"], "tealeaf" : ["small-tea", "big-tea"], "cufft" : ["80000000"],\
        "hpgmg-parallel": ["77", "88"], "tealeaf-parallel" : ["small-tea", "big-tea"], "cublas-long" : [16384], "gauss-seidel" : [10000000, 20000000, 22000000, 30000000]}
#sizes = {"cublas" : [16384], "linear" : [4194304], "random" : [4194304], "stream": [500002816]}
bsizes = [256]
#bsizes = [256, 2048]

dirs_fns = {"./vablock" : "*transfersv3*.png", 
            "./vablock-size" : "*transfer-sizev3*.png", 
            "./vablock-size-root" : "*transfer-sizev4*.png", 
            "./vablock-size-service" : "*transfer-sizev5*.png", 
            "./vablock-size-map" : "*transfer-sizev6*.png", 
            "./vablock-size-unmap" : "*transfer-sizev7*.png", 
            "./vablock-size-bt" : "*transfer-sizev8*.png", 
            "./vablock-size-makeres" : "*transfer-sizev9*.png", 
            "./vablock-size-respage" : "*transfer-sizev10*.png", 
            "./vablock-size-poppages" : "*transfer-sizev11*.png", 
            "./vablock-size-unmapres" : "*transfer-sizev12*.png", 
            "./vablock-size-unmaprdup" : "*transfer-sizev13*.png", 
            "./vablock-size-rmgetalloc" : "*transfer-sizev14*.png", 
            "./vablock-size-blockgpu" : "*transfer-sizev15*.png", 
            #"./vablock-size-mapcpu" : "*transfer-sizev16*.png", 
            #"./vablock-size-pmmmap" : "*transfer-sizev17*.png", 
            "./vablock-size-newvablock" : "*transfer-sizev18*.png", 
            "./vablock-size-unmaprescount" : "*transfer-sizev19*.png", 
            "./vablock-size-trans" : "*transfer-size-trans*.png", 
            "./vablock-size-pref" : "*transfer-size-pref*.png", 
            "./vablock-block-timeline" : "*batch-block-timeline*.png", 
            "./vablock-unmap-timeline" : "*batch-unmap-timeline*.png", 
            "./vablock-evict-timeline" : "*batch-evict-timeline*.png", 
            "./vablock-prefetch-timeline" : "*batch-prefetch-timeline*.png", 
            "./dups" : "*transfer*v2*.png", 
            "./evict" : "*evict*.png", 
            "./dist" : "*dist*.png",
            "./no-transfer" : "*no-transfer*.png"}

def clean_dirs(d):
    p = Path(d)
    if p.is_dir():
        shutil.rmtree(d)
    p.mkdir(parents=True, exist_ok=True)

import math
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
def all_size_plot():
    plt.rcParams.update({"text.usetex": True, 
                        "font.family": "sans-serif", 
                        "font.sans-serif": ["Helvetica"], 
                        "font.size": 18,
                        "legend.fontsize" : 16})

    myapps = [app for app in apps if "parallel" not in app and "long" not in app and "random" not in app and "tealeaf" not in app and "gauss" not in app]
    files = [f'../../benchmarks/{app}/log_{sizes[app][0]}_bsize_256/{app}_0.txt' for app in myapps]


    print(files)

    experiments = [Experiment(f) for f in files]
    for e, app in zip(experiments, myapps):
        #if "random" in app:
        #    continue
        xs = np.asarray(e.get_transfer_size()) #/ 1024
        ys = e.get_batch_times()
        xs, ys = zip(*sorted(zip(xs, ys), key=lambda x: x[0]))

        c = polyfit(xs, ys, 1)
        
        x_c = xs[0]
        ye = []
        li = []
        for i,x in enumerate(xs):
            if x != x_c:
                ye.append(li)
                x_c = x
                li = []
            li.append(ys[i])
        if li != []:
            ye.append(li)
        
        ye = [np.std(l) for l in ye]
        #print (ye)
            
        #The next line is again more elegant, but slower:
        #grouper = groupby(L, key=operator.itemgetter(0))
        #ye = [(x, np.var([yi for yi in y])) for x,y in grouper]
        #ye = [(x, [yi for yi in y]) for x,y in grouper]
        #print(ye)

        #p = plt.plot(xs, ys, '.', label=app)
        #p = plt.plot(xs, ys, '.', label=app)
        #colors = p[0].get_color()
        xs = np.asarray(sorted(set(xs)))
        if "cublas" in app:
            app = "sgemm"
        corr1 = plt.errorbar(xs, polyval(xs, c), yerr = ye, label=app, alpha=0.7)
        #corr1 = plt.errorbar(xs, m1 * np.asarray(xs) + b1, yerr= ye, color=colors, label=app)
        #plt.errorbar(year,T,yerr=dT,fmt='o',ms=5,color='black',alpha=0.75)
        
        
    #    plt.plot(xs, ys,  label=pp)

    plt.legend(loc="lower right")

    plt.title("Polynomial Fit for Batch Data")
    #plt.title("Least Squares Polynomial Fit for Batch Data")
    plt.xlabel("Batch Data Migration Size (KB)")
    plt.ylabel("Batch Execution Time (ns)")
    plt.ylim(0, 1e6)

    figname = "app-sizes.png"
    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname, dpi=500)
    plt.close()
    #sys.exit()

def main():

    # clean out prior images and directories
    for f in glob("*.png"):
        os.remove(f)
    for d in dirs_fns.keys():
        clean_dirs(d)
        clean_dirs(d + "-pf")
        clean_dirs(d + "-os")
        clean_dirs(d + "-pfos")


    all_size_plot()
    #sys.exit()

    procs = []
    for app in apps:
        print(f"generating figs for {app}")
        for size in sizes[app]:
            for bsize in bsizes:
                p = subprocess.Popen(['python3', 'batch_dist_plot.py',  f'../../benchmarks/{app}/log_{size}_bsize_{bsize}/{app}_0.txt'])
                procs.append(p)
                #p.wait()
                p = subprocess.Popen(['python3', 'batch_dist_plot.py',  f'../../benchmarks/{app}/log_pf_{size}_bsize_{bsize}/{app}_0.txt'])
                procs.append(p)
                #p.wait()
    while len(procs) > 0:
        procs.pop(0).wait() 
        #print(f"{app} figs done")


    for app in apps:
        for i, size in enumerate(humansorted(sizes[app])):
            for bsize in bsizes:
                base_str = f"{app}-0-log-pf-{size}-bsize-{bsize}-batch-size-time-"
                for d in dirs_fns.keys():
                    files = glob(base_str + dirs_fns[d])
                    print (base_str + dirs_fns[d])
                    if i > 0:
                        d = d + "-pfos"
                    else:
                        d = d + "-pf"
                    for f in files:
                        Path(f).rename(d + f"/{basename(f)}")

                base_str = f"{app}-0-log-{size}-bsize-{bsize}-batch-size-time-"
                for d in dirs_fns.keys():
                    files = glob(base_str + dirs_fns[d])
                    if i > 0:
                        d = d + "-os"
                    for f in files:
                        Path(f).rename(d + f"/{basename(f)}")



if __name__== "__main__":
  main()

