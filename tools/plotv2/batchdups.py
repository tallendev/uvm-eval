#!/usr/bin/python3
#import experimental_models
#from experimental_models import opts, metrics
import sys
import argparse
import sys
import os
import argparse
import statistics
from os.path import basename, splitext, dirname
from collections import Counter

import itertools, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import itertools
from multiprocessing import Pool

import Fault
from Fault import Experiment

NUM_THREADS=12

def plot_batch(x, y, batchnum, ranges, args):
    
    x = [hex(v) for v in x]
    # x address
    # y frequency
    
    plt.clf()
    print(x, y)
    plt.bar(x, y, width=0.5)
    for s in ranges:
        plt.plot([], [], label=s) 
    print("legend ranges", ranges)
    plt.legend(loc='upper right', title="Addr Ranges")
    plt.xlabel("Relative Fault Address")
    plt.ylabel("# of Occurrences")

    psize = basename(dirname(args.csv)) #.split("_")[-1]
    print ("psize:", psize)
    
    prefix = splitext(basename(args.csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    plt.title(f"{prefix} Batch {batchnum}")

    figname = None
    if args.o == "": 
        figname = "batchdups/" + (splitext(basename(args.csv))[0] + "-" + psize +  f"-batchdups-{batchnum}.png").replace("_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname, dpi=800)


def count_occurrences(numbers):
    unique_numbers = list(set(numbers))
    counts = [numbers.count(num) for num in unique_numbers]
    return unique_numbers, counts

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    parser.add_argument('-o', type=str, default="", help='output filename')
    args = parser.parse_args()
    c = args.csv
    m = "*"

    if not ".txt" in args.csv:
        print("Suspicious input file with no/wrong extension:", args.csv)
        exit(1)
    
    e = Experiment(c)
    
    ranges, faults, batches, num_batches = e.get_raw_data()
    e.print_info()

    matplotlib.rcParams['agg.path.chunksize'] = 10000
   

    fmin = min([f.fault_address for f in faults])
    ranges = [hex(r - fmin) for r in ranges]

    print("ranges:", ranges)
    iters = list(range(0, 50)) + list(range(50, len(batches), 10))
    #with Pool(processes=NUM_THREADS) as pool:
    for i in iters:
        b = batches[i]
        faults = sorted([f.fault_address - fmin for f in b])
        unique, counts = count_occurrences(faults)
        plot_batch(unique, counts, i, ranges, args)


if __name__== "__main__":
  main()

