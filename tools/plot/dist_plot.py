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
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(sm_ids)))
    
    id_f_map = {sm_id:[] for sm_id in sm_ids}
    for fault in faults:
        mapping = (fault.utlb_id, fault.client_id)
        id_f_map[mapping].append(fault)


    times_between_faults = []
    for sm_id in id_f_map.keys():
        for first, second in zip(sorted(id_f_map[sm_id][0:-1], key=lambda f: f.timestamp), sorted(id_f_map[sm_id][1:], key=lambda f: f.timestamp)):
            times_between_faults.append(second.timestamp - first.timestamp)
    avg = np.mean(times_between_faults)
    print("avg time between faults:", avg)
    print("min time between faults", min(times_between_faults))
    print("max time between faults", max(times_between_faults))
    
    
    print ("total # deltas:", len(times_between_faults))
    print ("Num faults < 1e1:", len([t for t in times_between_faults if t < 1e1]))
    print ("Num faults < 1e2:", len([t for t in times_between_faults if t < 1e2]))
    print ("Num faults < 1e3:", len([t for t in times_between_faults if t < 1e3]))
    print ("Num faults < 1e4:", len([t for t in times_between_faults if t < 1e4]))
    print ("Num faults < 1e5:", len([t for t in times_between_faults if t < 1e5]))
    print ("Num faults < 1e6:", len([t for t in times_between_faults if t < 1e6]))
    print ("Num faults < 1e7:", len([t for t in times_between_faults if t < 1e7]))
    print ("Num faults < 1e8:", len([t for t in times_between_faults if t < 1e8]))

    
    Q1 = np.quantile(times_between_faults, 0.25)
    Q3 = np.quantile(times_between_faults, 0.75)
    med = statistics.median(times_between_faults)

    print ("Q1, median, Q3:", Q1, ",", med, ",", Q3)


    #plt.plot(times, counts, marker="*")
    hist, bins, _ = plt.hist(times_between_faults, bins=16)
    
    print ("bins:", bins)
    logbins = np.logspace(0.0, np.log10(bins[-1]), len(bins))
    print ("logbins:", logbins)
    
    plt.clf()

    hist, bins, _ = plt.hist(times_between_faults, bins=logbins)
    
    ax = plt.gca()
    plt.vlines([Q1, med, Q3], 0, 1, transform=ax.get_xaxis_transform(), label="Q1/Med/Q3")

    plt.xlim(xmin=1e0)
    plt.ylim(ymin=0.0)
    
    plt.xscale("log")

    plt.xlabel("Time Between Same-SM Fault Arrival in Buffer (NS)")
    plt.ylabel("Frequency")

    plt.legend()


    psize = basename(dirname(args.csv)) #.split("_")[-1]
    print ("psize:", psize)

    figname = None
    if args.o == "": 
        figname = (splitext(basename(args.csv))[0] + "-" + psize +  "-sm-time-dist.png").replace("_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    #figname = splitext(basename(args.csv))[0].split("_")[0] + "-" + psize +  "-sm-time-dist.png"

    
    plt.tight_layout()
    print('saving figure:', figname)
    fig.savefig(figname, dpi=500)
    plt.close(fig)

if __name__== "__main__":
  main()

