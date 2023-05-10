#!/usr/bin/python3
#import experimental_models
#from experimental_models import opts, metrics
import sys
import argparse
import sys
import os
import argparse
import statistics
from colorsys import hsv_to_rgb
from os.path import basename, splitext, dirname
from collections import Counter, defaultdict

import itertools, matplotlib

import scipy.stats

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import itertools
from multiprocessing import Pool

import Fault
from Fault import Experiment

NUM_THREADS=44

# thanks gpt
def sort_lists(list1, list2):
    sorted_lists = sorted(zip(list1, list2), key=lambda x: x[0])
    sorted_list1 = [x[0] for x in sorted_lists]
    sorted_list2 = [x[1] for x in sorted_lists]
    return sorted_list1, sorted_list2

def count_occurrences(numbers):
    unique_numbers = list(set(numbers))
    counts = [numbers.count(num) for num in unique_numbers]
    return unique_numbers, counts

def count_occurrences_all(numbers):
    # using a defaultdict structure, use every number in numbers as a key, and store the count of each number where
    # the count is provided by number.num_instances
    d = defaultdict(int)
    for number in numbers:
        d[number.fault_address] += number.num_instances
    keys = d.keys()
    return sort_lists(keys, [d[val] for val in keys])

def count_occurrences_intra(numbers):
    # using a defaultdict structure, use every number in numbers as a key, and store the count of each number where
    # the count is provided by number.num_instances
    d = defaultdict(int)
    for number in numbers:
        d[number.fault_address] += number.num_instances - 1
    keys = d.keys()
    return sort_lists(keys, [d[val] for val in keys])


def process_batch_bar(i, b, fmin, ranges, args):
    faults = sorted([f.fault_address - fmin for f in b])
    unique, counts = count_occurrences(faults)
    outdir = "batchdups"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plot_batch_bar(unique, counts, i, ranges, args, outdir)

def process_batch_all_bar(i, b, fmin, ranges, args):
    faults = sorted(b, key=lambda f: f.fault_address)
    unique, counts = count_occurrences_all(faults)
    unique = [f - fmin for f in unique]
    outdir = "batchdupsall"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plot_batch_bar(unique, counts, i, ranges, args, outdir)

# chat gpt
def generate_colors(k):
    """
    Generates a list of k equally distant colors.
    """
    colors = []

    # Generate k evenly spaced hue values
    hue_values = [i / float(k) for i in range(k)]

    # Convert each hue value to an RGB color
    for hue in hue_values:
        r, g, b = hsv_to_rgb(hue, 0.6, 0.95)
        colors.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))

    # Return the list of equally distant colors
    return colors

def plot_batch_bar(x, y, batchnum, ranges, args, outdir):
    
    x = [hex(v) for v in x]
    # x address
    # y frequency
    
    plt.clf()

    #range_faultsx = [[] for _ in range(len(ranges))]
    #range_faultsy = [[] for _ in range(len(ranges))]
    #print(x, y)
    color_list = generate_colors(len(ranges))
    colors = []
    labels = []
    for val, valy in zip(x, y):
        for i,r in enumerate(ranges[::-1]):
            if val >= r:
                #range_faultsx[i].append(val)
                #range_faultsy[i].append(valy)
                colors.append(color_list[i])
                labels.append(r)
                break
    plt.bar(x, y, width=0.5, color=colors)

    for i, s in enumerate(ranges[::-1]):
        plt.plot([], [], color=color_list[i], label=s)
    #print("legend ranges", ranges)
    plt.legend(loc='upper right', title="Addr Ranges")
    plt.xlabel("Relative Fault Address")
    plt.ylabel("# of Occurrences")

    psize = basename(dirname(args.csv)) #.split("_")[-1]
    #print ("psize:", psize)
    
    prefix = splitext(basename(args.csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    plt.title(f"{prefix} Batch {batchnum}")

    # require the X axis labels to be listed vertically
    plt.xticks(rotation=90, fontsize='small')

    # shrink the x axis font in cases where the x axis has so many elements that the labels run together

    figname = None
    if args.o == "":
        full_outdir = f"{outdir}/{prefix}/"
        if not os.path.exists(full_outdir):
            os.makedirs(full_outdir)
        figname = full_outdir + (splitext(basename(args.csv))[0] + "-" + psize +  f"-batchdups-{batchnum}.png").replace("_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname)

    #for range in ranges:
    #    plt.clf()



def plot_dups_faults(x, y, args, outdir):
    # x dups
    # y faults

    plt.clf()

    #print(x, y)
    plt.scatter(x, y)
    #plt.legend(loc='upper right', title="Addr Ranges")
    plt.ylabel("# of Duplicates")
    plt.xlabel("# of Faults")

    psize = basename(dirname(args.csv))  # .split("_")[-1]
    #print("psize:", psize)

    prefix = splitext(basename(args.csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    plt.title(f"{prefix}")

    # require the X axis labels to be listed vertically
    plt.xticks(rotation=90, fontsize='small')

    # shrink the x axis font in cases where the x axis has so many elements that the labels run together

    figname = None
    if args.o == "":
        full_outdir = f"{outdir}/{prefix}/"
        if not os.path.exists(full_outdir):
            os.makedirs(full_outdir)
        figname = full_outdir + (splitext(basename(args.csv))[0] + "-" + psize + f"-dupfaults.png").replace(
            "_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname)


def plot_dups_faults_range(x, y, rangemin, args, outdir):
    # x dups
    # y faults

    plt.clf()

    #print("max: ", max(y))
    plt.scatter(x, y)
    #plt.legend(loc='upper right', title="Addr Ranges")
    plt.ylabel("# of Duplicates")
    plt.xlabel("# of Faults")

    psize = basename(dirname(args.csv))  # .split("_")[-1]
    print("psize:", psize)

    prefix = splitext(basename(args.csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    plt.title(f"{prefix}-{hex(rangemin)}")

    # require the X axis labels to be listed vertically
    plt.xticks(rotation=90, fontsize='small')

    # shrink the x axis font in cases where the x axis has so many elements that the labels run together

    figname = None
    if args.o == "":
        full_outdir = f"{outdir}/{prefix}/"
        if not os.path.exists(full_outdir):
            os.makedirs(full_outdir)
        figname = full_outdir + (splitext(basename(args.csv))[0] + "-" + psize + f"-dupfaults-{hex(rangemin)}.png").replace(
            "_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname)


def plot_dups_faults_cdf(x, y, args, outdir):
    # x dups
    # y faults

    plt.clf()

    #print(x, y)
    plt.plot(x, y)
    #plt.legend(loc='upper right', title="Addr Ranges")
    plt.xlabel("# of Duplicates")
    plt.ylabel("Probability")

    psize = basename(dirname(args.csv))  # .split("_")[-1]
    #print("psize:", psize)

    prefix = splitext(basename(args.csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    plt.title(f"{prefix}")

    # require the X axis labels to be listed vertically
    plt.xticks(rotation=90, fontsize='small')

    # shrink the x axis font in cases where the x axis has so many elements that the labels run together

    figname = None
    if args.o == "":
        full_outdir = f"{outdir}/{prefix}/"
        if not os.path.exists(full_outdir):
            os.makedirs(full_outdir)
        figname = full_outdir + (splitext(basename(args.csv))[0] + "-" + psize + f"-dupfaultscdf.png").replace(
            "_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname)


def plot_faults_cdf(x, y, args, outdir):
    # x dups
    # y faults

    plt.clf()

    #print(x, y)
    plt.plot(x, y)
    #plt.legend(loc='upper right', title="Addr Ranges")
    plt.xlabel("Fault Addr")
    plt.ylabel("Probability")

    psize = basename(dirname(args.csv))  # .split("_")[-1]
    #print("psize:", psize)

    prefix = splitext(basename(args.csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    plt.title(f"{prefix}")

    # require the X axis labels to be listed vertically
    plt.xticks(rotation=90, fontsize='small')

    # shrink the x axis font in cases where the x axis has so many elements that the labels run together

    figname = None
    if args.o == "":
        full_outdir = f"{outdir}/{prefix}/"
        if not os.path.exists(full_outdir):
            os.makedirs(full_outdir)
        figname = full_outdir + (splitext(basename(args.csv))[0] + "-" + psize + f"-faultscdf.png").replace(
            "_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname)


def plot_hist(x, args, xlabel, outdir, ext=""):
    # x dups
    # y faults

    plt.clf()
    plt.hist(x, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    #print(x, y)
    #plt.legend(loc='upper right', title="Addr Ranges")
    plt.xlabel(xlabel)

    psize = basename(dirname(args.csv))  # .split("_")[-1]
    #print("psize:", psize)

    prefix = splitext(basename(args.csv))[0].split('_')[0] + "-" + psize.split("_")[1]
    plt.title(f"{prefix}")

    # require the X axis labels to be listed vertically
    plt.xticks(rotation=90, fontsize='small')

    # shrink the x axis font in cases where the x axis has so many elements that the labels run together

    figname = None
    if args.o == "":
        full_outdir = f"{outdir}/{prefix}/"
        if not os.path.exists(full_outdir):
            os.makedirs(full_outdir)
        figname = full_outdir + (splitext(basename(args.csv))[0] + "-" + psize + f"-dupshist{ext}.png").replace(
            "_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname)


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

    plt.rcParams["figure.figsize"] = (18.5, 10.5)

    fmin = min([f.fault_address for f in faults])
    ranges_str = [hex(r - fmin) for r in ranges]
    #ranges = [r-fmin for r in ranges]

    print("ranges:", ranges)
    iters = list(range(0, 50)) + list(range(50, len(batches), 10))
    with Pool(processes=NUM_THREADS) as pool:
        #pool.starmap(process_batch_all_bar, [(i, batches[i], fmin, ranges_str, args) for i in iters])
        #pool.starmap(process_batch_bar, [(i, batches[i], fmin, ranges_str, args) for i in iters])
        process_dups_faults(batches, args, ranges.copy(), "dupfaultsinter", pool)
        process_alldups_faults(batches, args, ranges.copy(), "dupfaultsall", pool)
        process_intradups_faults(batches, args, ranges.copy(), "dupfaultsintra", pool)

def process_intradups_faults(batches, args, ranges, outdir, pool):
    ranges.append(sys.maxsize)
    bfaults = []
    bdups = []
    faultlist = []
    for b in batches:
        faults = [f.fault_address for f in b]
        faultlist += faults
        _, counts = count_occurrences_all(b)
        # 4/25; updated the # of faults to include intra
        bfaults.append(sum(counts))
        unique, counts = count_occurrences_intra(b)
        counts = [c for c in counts]
        #counts = [c-1 for c in counts]
        #print(f"{sum(counts)}, {len(b)}")
        bdups.append(sum(counts))
        #bfaults.append(len(b))
    # dups

    fmin = min(faultlist)

    plot_dups_faults(bfaults, bdups, args, outdir)
    # TODO you parallelized this the wrong way bozo go fix it
    for i in range(len(ranges) - 1):
        rbatches = []
        rbfaults = []
        rbdups = []
        for b in batches:
            for j in range(len(b)):
                batch = []
                for fault in b:
                    if ranges[i] <= fault.fault_address < ranges[i + 1]:
                        batch.append(fault)
                if batch:
                    rbatches.append(b)
        if (rbatches):
            for b in rbatches:
                # 4/25; updated the # of faults to include intra
                _, counts = count_occurrences_all(b)
                rbfaults.append(sum(counts))
                unique, counts = count_occurrences_intra(b)
                counts = [c for c in counts]
                #counts = [c - 1 for c in counts]
                rbdups.append(sum(counts))
                #rbfaults.append(len(b))
        r = ranges[i] - fmin
        process_hist(args, rbdups, outdir, f"Intra-SM Duplicates - {r}", f"-{hex(r)}")
        plot_dups_faults_range(rbfaults, rbdups, ranges[i] - fmin, args, outdir)
        #pool.starmap(process_range_batch, [(b, i, ranges, fmin, args, outdir) for b in batches])

    bdups.sort()
    #cdf = scipy.stats.norm.cdf(bdups)

    process_hist(args, bdups, outdir, f"Intra-SM Duplicates")


def process_range_batch_all(b, i, ranges, fmin, args, outdir):
    rbatches = []
    rbfaults = []
    rbdups = []
    for j in range(len(b)):
        batch = []
        for fault in b:
            if ranges[i] <= fault.fault_address < ranges[i+1]:
                batch.append(fault)
        if batch:
            rbatches.append(b)
    if (rbatches):
        for b in rbatches:
            unique, counts = count_occurrences_all(b)
            counts = [c - 1 for c in counts]
            rbdups.append(sum(counts))
            rbfaults.append(len(b))
    plot_dups_faults_range(rbfaults, rbdups, ranges[i]-fmin, args, outdir)



def process_alldups_faults(batches, args, ranges, outdir, pool):
    ranges.append(sys.maxsize)
    bfaults = []
    bdups = []
    faultlist = []
    for b in batches:
        faults = [f.fault_address for f in b]
        faultlist += faults
        unique, counts = count_occurrences_all(b)
        # 4/25; updated the # of faults to include intra
        bfaults.append(sum(counts))
        counts = [c-1 for c in counts]
        #print(counts)
        bdups.append(sum(counts))
        #bfaults.append(len(b))
    plot_dups_faults(bfaults, bdups, args, outdir)

    fmin = min(faultlist)

    for i in range(len(ranges) - 1):
        rbatches = []
        rbfaults = []
        rbdups = []
        for b in batches:
            for j in range(len(b)):
                batch = []
                for fault in b:
                    if ranges[i] <= fault.fault_address < ranges[i + 1]:
                        batch.append(fault)
                if batch:
                    rbatches.append(b)
        if (rbatches):
            for b in rbatches:
                unique, counts = count_occurrences_all(b)
                # 4/25; updated the # of faults to include intra
                rbfaults.append(sum(counts))
                counts = [c - 1 for c in counts]
                rbdups.append(sum(counts))
                #rbfaults.append(len(b))
        r = ranges[i] - fmin
        plot_dups_faults_range(rbfaults, rbdups, r, args, outdir)
        process_hist(args, rbdups, outdir, f"All-SM Duplicates - {r}", f"-{hex(r)}")

    process_hist(args, bdups, outdir, "All-SM Duplicates")
    #faultlist = sorted(set(faultlist))
    #faultlist = [f - fmin for f in faultlist]
    #cdf = scipy.stats.norm.cdf(faultlist)
    #plot_faults_cdf(faultlist, cdf, args, outdir)


def process_range_batch(b, i, ranges, fmin, args, outdir):
    rbatches = []
    rbfaults = []
    rbdups = []
    for j in range(len(b)):
        batch = []
        for fault in b:
            if ranges[i] <= fault.fault_address < ranges[i+1]:
                batch.append(fault)
        if batch:
            rbatches.append(b)
    if (rbatches):
        for b in rbatches:
            faults = [f.fault_address for f in b]
            unique, counts = count_occurrences(faults)
            counts = [c - 1 for c in counts]
            rbdups.append(sum(counts))
            rbfaults.append(len(b))
    plot_dups_faults_range(rbfaults, rbdups, ranges[i]-fmin, args, outdir)



def process_dups_faults(batches, args, ranges, outdir, pool):
    ranges.append(sys.maxsize)
    bfaults = []
    bdups = []
    faultlist = []
    for b in batches:
        faults = [f.fault_address for f in b]
        faultlist += faults
        _, counts = count_occurrences_all(b)
        # 4/25; updated the # of faults to include intra
        bfaults.append(sum(counts))
        unique, counts = count_occurrences(faults)
        counts = [c-1 for c in counts]
        #print(f"{sum(counts)}, {len(b)}")
        bdups.append(sum(counts))
        #bfaults.append(len(b))
    # dups

    fmin = min(faultlist)

    plot_dups_faults(bfaults, bdups, args, outdir)
    # TODO you parallelized this the wrong way bozo go fix it
    for i in range(len(ranges) - 1):
        rbatches = []
        rbfaults = []
        rbdups = []
        for b in batches:
            for j in range(len(b)):
                batch = []
                for fault in b:
                    if ranges[i] <= fault.fault_address < ranges[i + 1]:
                        batch.append(fault)
                if batch:
                    rbatches.append(b)
        if (rbatches):
            for b in rbatches:
                # 4/25; updated the # of faults to include intra
                _, counts = count_occurrences_all(b)
                rbfaults.append(sum(counts))
                unique, counts = count_occurrences(b)
                counts = [c - 1 for c in counts]
                rbdups.append(sum(counts))
                #rbfaults.append(len(b))
        r = ranges[i] - fmin
        process_hist(args, rbdups, outdir, f"Inter-SM Duplicates - {r}", f"-{hex(r)}")
        plot_dups_faults_range(rbfaults, rbdups, r, args, outdir)
        #pool.starmap(process_range_batch, [(b, i, ranges, fmin, args, outdir) for b in batches])

    bdups.sort()
    #cdf = scipy.stats.norm.cdf(bdups)

    process_hist(args, bdups, outdir, "Inter-SM Duplicates")

    #cdf = []
    #s = sum(bdups)
    #for i, val in enumerate(bdups):
    #    cdf.append(sum(bdups[0:i+1])/bdups)
    # faults
    #faultlist = sorted(set(faultlist))
    #faultlist = sorted(faultlist)
    #faultlist = [f - fmin for f in faultlist]
    #cdf = scipy.stats.norm.cdf(faultlist)
    #plot_faults_cdf(faultlist, cdf, args, outdir)
    # per-range cdf-


def process_hist(args, bdups, outdir, label, ext=""):
    bdups_data = Counter(bdups)
    bdups_keys = np.sort(list(bdups_data.keys()))
    bdups_count = np.array([bdups_data[key] for key in bdups_keys])
    cdf = np.cumsum(bdups_count) / len(bdups)
    print("bdups:", *bdups)
    print("cdf:", *cdf)
    plot_dups_faults_cdf(bdups_keys, cdf, args, outdir)
    plot_hist(bdups, args, label, outdir, ext)



if __name__== "__main__":
  main()

