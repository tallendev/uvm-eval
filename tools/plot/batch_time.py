#!/usr/bin/python3
#import experimental_models
#from experimental_models import opts, metrics
import sys
import argparse
import sys
import os
import argparse
from os.path import basename, splitext, dirname

import itertools, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import itertools

import Fault
from Fault import Experiment

marker="."
markersize=1

apps = ["cublas", "linear", "random", "stream"]
sizes = {"cublas" : [16384], "linear" : [4194304], "random" : [4194304], "stream": [500002816]}
bsizes = [256]

def do_all_more(batches, faults, csv, x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5):

    figsize = plt.rcParams.get('figure.figsize')
    print("figsize:", figsize)
    figsize[1] = figsize[1] * 1.5

    plt.rcParams['figure.figsize'] = figsize
    fig, axs = plt.subplots(3,2)
    plt.xlabel("Time")
    axs[0, 0].plot(x0, y0, "b"+marker, markersize=markersize) 
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Batch Size")

    axs[0, 1].plot(x1, y1, "b"+marker, markersize=markersize) 
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Batch Size Unique 4K")

    axs[1, 0].plot(x2, y2, "b"+marker, markersize=markersize) 
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Batch Size Unique 64K")

    axs[1, 1].plot(x3, y3, "b"+marker, markersize=markersize) 
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Batch Size Unique 2MB")

    axs[2, 0].plot(x4, y4, "b"+marker, markersize=markersize) 
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].set_ylabel("# Transfers")

    axs[2, 1].plot(x5, y5, "b"+marker, markersize=markersize) 
    axs[2, 1].set_xlabel("Time")
    axs[2, 1].set_ylabel("# Transfers PF")
    

    figend = "batchsizes6"
    psize = None
    if "pf" in csv:
        psize = dirname(csv).split("_")[2]
    else:
        psize = dirname(csv).split("_")[1]


    figname = splitext(basename(csv))[0].split("_")[0] + "-" + psize + "-" + figend + ".png"
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    #fig.savefig(figname, dpi=500, figsize=figsize)
    plt.close(fig)


def do_all(batches, faults, csv, x0, x1, x2, x3, y0, y1, y2, y3):
    fig, axs = plt.subplots(2,2)
    plt.xlabel("Time")
    axs[0, 0].plot(x0, y0, "b"+marker, markersize=markersize) 
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Batch Size")

    axs[0, 1].plot(x1, y1, "b"+marker, markersize=markersize) 
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Batch Size Unique 4K")

    axs[1, 0].plot(x2, y2, "b"+marker, markersize=markersize) 
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Batch Size Unique 64K")

    axs[1, 1].plot(x3, y3, "b"+marker, markersize=markersize) 
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Batch Size Unique 2MB")
    

    figend = "batchsizessquare"
    psize = None
    if "pf" in csv:
        psize = dirname(csv).split("_")[2]
    else:
        psize = dirname(csv).split("_")[1]


    figname = splitext(basename(csv))[0].split("_")[0] + "-" + psize + "-" + figend + ".png"
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)


def do_batch_len_time_transfers_4k(batches, faults, csv):
    return do_batch_len_time_transfers(batches, faults, csv, 4096, "transf4k")

def do_batch_len_time_transfers_64k(batches, faults, csv):
    return do_batch_len_time_transfers(batches, faults, csv, 65536, "transf64k")

def do_batch_len_time_transfers(batches, faults, csv, div, divstr):
    total_xs = []

    x_ts = []
    y_sizes = []
    
    for batch in batches:
        vals = sorted(set([f.fault_address // div for f in batch]))
        transfers = 0
        prev = vals[0]
        count = 0
        for val in vals:
            if  val > 1 + prev or val % (2097152//div) == 0:
                transfers += 1
                count = 0
            else:
                count += 1
            prev = val
        if count > 0:
            transfers += 1

        y_sizes.append(transfers)
        x_ts.append(min(batch, key=lambda f: f.timestamp).timestamp)
    start_time = min(faults, key=lambda f: f.timestamp).timestamp
    x_ts = [f - start_time for f in x_ts]

    plot(x_ts, y_sizes, "Time", "# Transfers", divstr, csv)
    return (x_ts, y_sizes)


def do_batch_len_time_4k(batches, faults, csv):
    total_xs = []

    x_ts = []
    y_sizes = []
    
    for batch in batches:
        y_sizes.append(len(set([f.fault_address // 4096 for f in batch])))
        x_ts.append(min(batch, key=lambda f: f.timestamp).timestamp)
    start_time = min(faults, key=lambda f: f.timestamp).timestamp
    x_ts = [f - start_time for f in x_ts]

    plot(x_ts, y_sizes, "Time", "Batch Size", "batchsizes4k", csv)
    return (x_ts, y_sizes)


def do_batch_len_time_64k(batches, faults, csv):
    total_xs = []

    x_ts = []
    y_sizes = []
    
    for batch in batches:
        y_sizes.append(len(set([f.fault_address // 65536 for f in batch])))
        x_ts.append(min(batch, key=lambda f: f.timestamp).timestamp)
    start_time = min(faults, key=lambda f: f.timestamp).timestamp
    x_ts = [f - start_time for f in x_ts]

    plot(x_ts, y_sizes, "Time", "Batch Size", "batchsizes64k", csv)
    return (x_ts, y_sizes)

def do_batch_len_time_2m(batches, faults, csv):
    total_xs = []

    x_ts = []
    y_sizes = []
    
    for batch in batches:
        y_sizes.append(len(set([f.fault_address // 2097152 for f in batch])))
        x_ts.append(min(batch, key=lambda f: f.timestamp).timestamp)
    start_time = min(faults, key=lambda f: f.timestamp).timestamp
    x_ts = [f - start_time for f in x_ts]

    plot(x_ts, y_sizes, "Time", "Batch Size", "batchsizes2m", csv)
    return (x_ts, y_sizes)
    
def do_batch_len_time(batches, faults, csv):
    total_xs = []

    x_ts = []
    y_sizes = []
    
    for batch in batches:
        y_sizes.append(len(batch))
        x_ts.append(min(batch, key=lambda f: f.timestamp).timestamp)
    start_time = min(faults, key=lambda f: f.timestamp).timestamp
    x_ts = [f - start_time for f in x_ts]

    plot(x_ts, y_sizes, "Time", "Batch Size", "batchsizes", csv)
    return (x_ts, y_sizes)

def pages_per_vablock_per_batch(batches, faults):
    vablock_pages = []
    for batch in batches:
        vablocks = dict()
        for f in batch:
            vablock = f.fault_address // 2097152
            page = f.fault_address // 4096
            if vablock in vablocks:
                vablocks[vablock].append(page)
            else:
                vablocks[vablock] = [page]
        vablock_pages.append(vablocks)

    faults_per_vablock = []
    faults_per_vablock_nodup = []
    num_vablocks = []
    for d in vablock_pages:
        num_vablocks.append(len(d.keys()))
        for k in d.keys():
            faults_per_vablock.append(len(d[k]))
            faults_per_vablock_nodup.append(len(set(d[k])))

    print ("avg batch len:", np.mean([len(b) for b in batches]))
    print ("stddev batch len:", np.std([len(b) for b in batches]))

    print ("avg vablocks per batch:", np.mean(num_vablocks))
    print ("stddev vablocks per batch:", np.std(num_vablocks))

    print ("avg faults per vablock:", np.mean(faults_per_vablock))
    print ("stddev faults per vablock:", np.std(faults_per_vablock))
    print ("avg faults per vablock nodup:", np.mean(faults_per_vablock_nodup))
    print ("stddev faults per vablock nodup:", np.std(faults_per_vablock_nodup))


def plot(x, y, xlab, ylab, figend, csv):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if "pf" in csv:
        psize = dirname(csv).split("_")[2]
    else:
        psize = dirname(csv).split("_")[1]
    figname = splitext(basename(csv))[0].split("_")[0] + "-" + psize + "-" + figend + ".png"

    ax.plot(x, y, "b" + marker, markersize=markersize)

    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

def per_app(c):
 
    e = Experiment(c)
    
    ranges, faults, batches, num_batches = e.get_raw_data()
    e.print_info()

    matplotlib.rcParams['agg.path.chunksize'] = 10000
    
    x0, y0 = do_batch_len_time(batches, faults, c)
    x1, y1 = do_batch_len_time_4k(batches, faults, c)
    x2, y2 = do_batch_len_time_64k(batches, faults, c)
    x3, y3 = do_batch_len_time_2m(batches, faults, c)
    do_all(batches, faults, c, x0, x1, x2, x3, y0, y1, y2, y3)
    x4, y4 = do_batch_len_time_transfers_4k(batches, faults, c)
    x5, y5 = do_batch_len_time_transfers_64k(batches, faults, c)
    do_all_more(batches, faults, c, x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5)
    print(c)
    pages_per_vablock_per_batch(batches, faults)

def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('csv', type=str, help='Full path to CSV file')
        args = parser.parse_args()
        c = args.csv

        if not ".txt" in args.csv:
            print("Suspicious input file with no/wrong extension:", args.csv)
            exit(1)
        per_app(c)
    else:
        print ("Processing defaults")
        for app in apps:
            for size in sizes[app]:
                for bsize in bsizes:
                    c = f"../../benchmarks/{app}/log_{size}_bsize_{bsize}/{app}_0.txt"
                    print(c)
                    per_app(c)

if __name__== "__main__":
  main()

