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

def size_time_plot(xs, ys, psize, args, suffix):
    #TODO assume xs and ys are same order
    xs, ys = zip(*sorted(zip(xs, ys), key=lambda t: t[0]))
    plt.plot(xs, ys, "b.")
    plt.xlabel("Batch Size")
    plt.ylabel("Batch Time")


    if args.o == "": 
        if len(suffix) > 0:
            figname = (args.d + "/" + splitext(basename(args.csv))[0] + "-" + psize +  f"-batch-size-time-{suffix}.png").replace("_", "-")
        else:
            figname = (args.d + "/" + splitext(basename(args.csv))[0] + "-" + psize +  f"-batch-size-time.png").replace("_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"
    
    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname, dpi=500)
    plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    parser.add_argument('-o', type=str, default="", help='output filename')
    parser.add_argument('-d', type=str, default="", help='out dir')
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
    #fig = plt.gcf()
    #ax = fig.add_subplot(1,1,1)
    #ax2 = ax.twiny()
    
    read=[]
    rx = []
    write=[]
    wx = []
    for i,f in enumerate(faults):
        if f.access_type == "r":
            #read.append(f.fault_address)
            #rx.append(i)
            pass
        elif f.access_type == "w":
            #write.append(f.fault_address)
            #wx.append(i)
            pass
        elif f.access_type == "p":
            pass
        else:
            print("unaccounted for access type:", f.access_type)

    true_bl = [len(batch) for batch in batches]
    bl = [0] + true_bl
    for i,b in enumerate(bl):
        if i == 0:
            continue
        bl[i] += bl[i-1]
    del bl[0]

    #print(batches)
    
    sm_ids = sorted(e.count_utlb_client_pairs().keys())
    print ("sm_ids:", sm_ids)
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(sm_ids)))
    
    id_f_map = {sm_id:[] for sm_id in sm_ids}
    for fault in faults:
        mapping = (fault.utlb_id, fault.client_id)
        id_f_map[mapping].append(fault)


    times_between_batches = e.batch_times
    avg = e.avg_time_batches()
    print("avg time between batches:", avg)
    print("min time between batches", min(times_between_batches))
    print("max time between batches", max(times_between_batches))
    
    
    print ("total # deltas:", len(times_between_batches))
    print ("Num batches < 1e1:", len([t for t in times_between_batches if t < 1e1]))
    print ("Num batches < 1e2:", len([t for t in times_between_batches if t < 1e2]))
    print ("Num batches < 1e3:", len([t for t in times_between_batches if t < 1e3]))
    print ("Num batches < 1e4:", len([t for t in times_between_batches if t < 1e4]))
    print ("Num batches < 1e5:", len([t for t in times_between_batches if t < 1e5]))
    print ("Num batches < 1e6:", len([t for t in times_between_batches if t < 1e6]))
    print ("Num batches < 1e7:", len([t for t in times_between_batches if t < 1e7]))
    print ("Num batches < 1e8:", len([t for t in times_between_batches if t < 1e8]))

    
    Q1 = np.quantile(times_between_batches, 0.25)
    Q3 = np.quantile(times_between_batches, 0.75)
    med = statistics.median(times_between_batches)
    avg = np.mean(times_between_batches)

    print ("Q1, median, Q3:", Q1, ",", med, ",", Q3)


    #plt.plot(times, counts, marker="*")
    hist, bins, _ = plt.hist(times_between_batches)
    #hist, bins, _ = plt.hist(times_between_batches, bins=16)
    binlen = len(bins)
    
    print ("bins:", bins)
    logbins = np.logspace(0.0, np.log10(bins[-1]), len(bins))
    print ("logbins:", logbins)
    
    plt.clf()

    hist, bins, _ = plt.hist(times_between_batches, bins=logbins)
    
    ax = plt.gca()
    plt.vlines([Q1, med, Q3], 0, 1, transform=ax.get_xaxis_transform(), label="Q1/Med/Q3")
    plt.vlines([avg], 0, 1, transform=ax.get_xaxis_transform(), label="Avg", color="r")

    plt.xlim(xmin=1e0)
    plt.ylim(ymin=0.0)
    
    plt.xscale("log")

    plt.xlabel("Time Between Batch Fault Arrival in Buffer (NS)")
    plt.ylabel("Frequency")

    plt.legend()

    psize = basename(dirname(args.csv)) #.split("_")[-1]
    print ("psize:", psize)

    figname = None
    if args.o == "": 
        figname = (args.d + "/" + splitext(basename(args.csv))[0] + "-" + psize +  "-batch-time-dist.png").replace("_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    #figname = splitext(basename(args.csv))[0].split("_")[0] + "-" + psize +  "-sm-time-dist.png"
    
    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname, dpi=500)

    plt.close()
    
    if ("pf") in args.csv:
        xs = [len(batch) - dup + len(pfbatch) for batch, dup, pfbatch in zip(e.batches, e.get_duplicate_faults_64k(), e.pfbatches)]
    else:
        xs = [len(batch) - dup for batch, dup in zip(e.batches, e.get_duplicate_faults_4k())]
        #xs = [len(batch) - e.get_duplicate_faults_4k() for batch in e.batches]
    
    hist, bins, _ = plt.hist(xs, bins=len(logbins))
    logbins = np.logspace(0.0, np.log10(bins[-1]), len(bins))
    plt.clf()
    hist, bins, _ = plt.hist(xs, bins=logbins)


    #plt.hist([len(batch) for batch in e.batches])
    plt.xlabel("Batch Sizes")
    plt.ylabel("Frequency")
    
    plt.xlim(xmin=1e0)
    plt.ylim(ymin=0.0)
    plt.xscale("log")

    if args.o == "": 
        figname = (args.d + "/" + splitext(basename(args.csv))[0] + "-" + psize +  "-batch-size-dist.png").replace("_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname, dpi=500)

    plt.close()

    ys = times_between_batches

    size_time_plot(xs, ys, psize, args, "pf-dups")

    if ("pf") in args.csv:
        xs = [len(batch) - dup for batch, dup in zip(e.batches, e.get_duplicate_faults_64k())]
    else:
        xs = [len(batch) - dup for batch, dup in zip(e.batches, e.get_duplicate_faults_4k())]
        #xs = [len(batch) - e.get_duplicate_faults_4k() for batch in e.batches]
    size_time_plot(xs, ys, psize, args, "dups")

    if ("pf") in args.csv:
        xs = [len(batch) for batch in e.batches]
    else:
        xs = [len(batch) for batch in e.batches]
    
    size_time_plot(xs, ys, psize, args, "")

    div = 65536 if "pf" in args.csv else 4096
    xs = []
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

        xs.append(transfers)
    
    size_time_plot(xs, ys, psize, args, "transfers")



if __name__== "__main__":
  main()

