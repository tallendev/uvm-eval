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

import Fault
from Fault import Experiment


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
    fig = plt.figure()
    #fig = plt.gcf()
    #ax = fig.add_subplot(1,1,1)
    #ax2 = ax.twiny()
    
    #print(batches)
    
    
    
    plt.clf()

    
    ax = plt.gca()


    x = []
    y = []
    for i, batch in enumerate(batches):
        y_set = set()
        for fault in batch:
            if fault.utlb_id not in y_set:
                x.append(i)
                y_set.add(fault.utlb_id)

        y += [y for y in y_set]
            
    plt.plot(x, y, "bo", markersize=1)
        
    print("max tlb:", max(y))
    print("unique tlb", len(set(y)))

    plt.xlabel("Batch ID")
    plt.ylabel("UTLB Fault Present")

    #plt.legend()


    psize = basename(dirname(args.csv)) #.split("_")[-1]
    print ("psize:", psize)

    figname = None
    if args.o == "": 
        figname = (splitext(basename(args.csv))[0] + "-" + psize +  "-utlb-active.png").replace("_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    #figname = splitext(basename(args.csv))[0].split("_")[0] + "-" + psize +  "-sm-time-dist.png"

    
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=800)
    plt.close(fig)

if __name__== "__main__":
  main()

