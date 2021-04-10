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

from numpy.polynomial.polynomial import polyfit

import itertools, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import itertools

import Batch
from Batch import Experiment

from multiprocessing import Process

plt.rcParams.update({"font.family": "sans-serif", 
                    "font.sans-serif": ["Helvetica"], 
                    "font.size": 16

                    })


def size_time_plot_color_sub(xs, ys, cs, psize, args, suffix, clabel, cmap="BuGn", legend=True, poly=True, xlabel="Batch Data Migration Size (KB)"):
#"text.usetex": True

    #TODO assume xs and ys are same order
    xs, ys, cs= zip(*sorted(zip(xs, ys, cs), key=lambda t: t[0]))
    plt.scatter(xs, ys,  c=cs, marker='.', cmap=cmap, label="batch", edgecolors='k', linewidth=.1)
    #plt.scatter(xs, ys,  c=cs, marker='.', cmap="cool", label="batch")
    #plt.plot(xs, ys, "b.", c=cs, cmap=plt.get_cmap("viridis"), label="batch")

    if poly:
        b, m = polyfit(xs, ys, 1)
        plt.plot(xs, b + m * np.asarray(xs), '-', label=f"{b:.2f} + {m:.2f}x")

    if legend:
        plt.legend()
    plt.colorbar(label=clabel)

    plt.xlabel(xlabel)
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

    plt.rcParams.update({"font.family": "sans-serif", 
                        "font.sans-serif": ["Helvetica"], 
                        "font.size": 18

                        })
    
    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname, dpi=500)
    plt.close()

def size_time_plot_color(xs, ys, cs, psize, args, suffix, clabel, cmap="BuGn", legend=False, poly=True, xlabel="Batch Data Migration Size (KB)"):
    p = Process(target=size_time_plot_color_sub, args=(xs, ys, cs, psize, args, suffix, clabel, cmap, legend, poly, xlabel))
    p.start()
    return p

def size_time_plot(xs, ys, psize, args, suffix):
    #TODO assume xs and ys are same order
    xs, ys = zip(*sorted(zip(xs, ys), key=lambda t: t[0]))
    plt.plot(xs, ys, "b.", label="batch")

    b, m = polyfit(xs, ys, 1)
    plt.plot(xs, b + m * np.asarray(xs), '-', label=f"{b:.2f} + {m:.2f}x")

    plt.legend()

    plt.xlabel("Batch Data Migration Size (KB)")
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
    parser.add_argument('-d', type=str, default=".", help='out dir')
    args = parser.parse_args()
    c = args.csv
    m = "*"

    if not ".txt" in args.csv:
        print("Suspicious input file with no/wrong extension:", args.csv)
        exit(1)
    
    e = Experiment(c)

    matplotlib.rcParams['agg.path.chunksize'] = 10000

    batch_times = e.get_batch_times()
    
    Q1 = np.quantile(batch_times, 0.25)
    Q3 = np.quantile(batch_times, 0.75)
    med = statistics.median(batch_times)
    avg = np.mean(batch_times)

    print ("Q1, median, Q3:", Q1, ",", med, ",", Q3)


    #plt.plot(times, counts, marker="*")
    hist, bins, _ = plt.hist(batch_times)
    #hist, bins, _ = plt.hist(times_between_batches, bins=16)
    binlen = len(bins)
    
    print ("bins:", bins)
    logbins = np.logspace(0.0, np.log10(bins[-1]), len(bins))
    print ("logbins:", logbins)
    
    plt.clf()

    hist, bins, _ = plt.hist(batch_times, bins=logbins)
    
    ax = plt.gca()
    plt.vlines([Q1, med, Q3], 0, 1, transform=ax.get_xaxis_transform(), label="Q1/Med/Q3")
    plt.vlines([avg], 0, 1, transform=ax.get_xaxis_transform(), label="Avg", color="r")

    plt.xlim(xmin=1e0)
    plt.ylim(ymin=0.0)
    
    plt.xscale("log")

    plt.xlabel("Batch Times")
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
    
    xs = e.get_transfers()
    
    hist, bins, _ = plt.hist(xs, bins=len(logbins))
    logbins = np.logspace(0.0, np.log10(bins[-1]), len(bins))
    plt.clf()
    hist, bins, _ = plt.hist(xs, bins=logbins)


    #plt.hist([len(batch) for batch in e.batches])
    plt.xlabel("Batch Data Migration Size (KB)")
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

    xs = e.get_transfer_size()
    
    hist, bins, _ = plt.hist(xs, bins=len(logbins))
    logbins = np.logspace(0.0, np.log10(bins[-1]), len(bins))
    plt.clf()
    hist, bins, _ = plt.hist(xs, bins=logbins)


    #plt.hist([len(batch) for batch in e.batches])
    plt.xlabel("Batch Transfer Sizes")
    plt.ylabel("Frequency")
    
    plt.xlim(xmin=1e0)
    plt.ylim(ymin=0.0)
    plt.xscale("log")

    if args.o == "": 
        figname = (args.d + "/" + splitext(basename(args.csv))[0] + "-" + psize +  "-batch-tsize-dist.png").replace("_", "-")
    else:
        figname = args.o
        if ".png" not in figname:
            figname += ".png"

    plt.tight_layout()
    print('saving figure:', figname)
    plt.savefig(figname, dpi=500)

    plt.close()

    ps = []

    xs = e.get_batch_relative_start_times()
    ys = e.get_batch_times()
    cs = e.get_block_gpu_state_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "batch-block-timeline", "Block State % Time", poly=False, xlabel="Timeline (ns)") #, cmap="plasma")
    ps.append(p)

    xs = e.get_batch_relative_start_times()
    ys = e.get_batch_times()
    cs = e.get_unmap_res_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "batch-unmap-timeline", "Unmapping % Time", poly=False, xlabel="Timeline (ns)") #, cmap="plasma")
    ps.append(p)

    cs = e.num_true_prefetches()
    if len(cs) > 0:
        xs = e.get_batch_relative_start_times()
        ys = e.get_batch_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "batch-prefetch-timeline", "# Prefetches", poly=False, xlabel="Timeline (ns)") #, cmap="plasma")
        ps.append(p)
    

    cs = e.get_evict_ops()
    if len(cs) > 0:
        xs = e.get_batch_relative_start_times()
        ys = e.get_batch_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "batch-evict-timeline", "Evictions", poly=False, xlabel="Timeline (ns)") #, cmap="plasma")
        ps.append(p)




    xs = e.get_total_faults()
    ys = e.get_batch_times()
    cs = e.get_map_ops()
    

    p = size_time_plot_color(xs, ys, cs, psize, args, "", "\# Map Ops")
    ps.append(p)
    plt.xlabel("Batch Data Migration Size (KB)")
    plt.ylabel("Batch Time")

    xs = e.get_transfers() 
    cs = e.get_map_ops()
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfers", "\# Mapping Ops")
    ps.append(p)
    cs = [c - t for t, c in zip(e.get_transfers(), e.get_total_faults())]
    #cs = [c for c in e.get_total_faults()]
    #cs = [c for c in e.num_already_resident()]
    #cs = [c for c in e.get_num_mapped_pages()]
    #cs = [c for c in e.get_trimmed_faults()]
    if False:
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv2", "Duplicates")
        ps.append(p)
        cs = e.get_num_vablocks()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv3", "VABlocks")
        ps.append(p)
        cs = e.get_root_chunk_allocs()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv4", "Root Chunk Allocs")
        ps.append(p)

        cs = e.get_service_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv5", "Service Time")
        ps.append(p)

        cs = e.get_map_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv6", "Map Time")
        ps.append(p)
        
        cs = e.get_unmap_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv7", "Unmap Time")
        ps.append(p)

        cs = e.get_transfer_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv8", "Batch Transfer Time")
        ps.append(p)

        cs = e.get_make_resident_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv9", "Make Resident time")
        ps.append(p)
        
        cs = e.get_resident_pages_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv10", "Resident Pages Time")
        ps.append(p)

        cs = e.get_pop_pages_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv11", "Resident Pages Time")
        ps.append(p)
        
        cs = e.get_unmap_res_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv12", "Unmap Res Time", poly=False)
        ps.append(p)
        
        cs = e.get_unmap_rdup_times()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv13", "Unmap Rdup Time")
        ps.append(p)

        cs = e.get_rmask_get_allocs()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfersv14", "RMask Get Alloc Time")
        ps.append(p)

    xs = e.get_transfer_size() 
    #cs = e.get_map_ops()
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-size", "\# Mapping Ops")
    #ps.append(p)

    cs = [c - t for t, c in zip(e.get_transfers(), e.get_total_faults())]
    #cs = [c for c in e.get_total_faults()]
    #cs = [c for c in e.num_already_resident()]
    #cs = [c for c in e.get_num_mapped_pages()]
    #cs = [c for c in e.get_trimmed_faults()]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev2", "Duplicates")
    ps.append(p)
    cs = e.get_num_vablocks()
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev3", "VABlocks")
    ps.append(p)
    cs = e.get_root_chunk_allocs()
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev4", "Root Chunk Allocs")
    ps.append(p)
    
    cs = e.get_service_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev5", "Service % Time")
    ps.append(p)

    cs = e.get_map_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev6", "Map % Time")
    ps.append(p)

    cs = e.get_unmap_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev7", "Unmap % Time")
    ps.append(p)


    cs = e.get_transfer_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev8", "Batch Transfer % Time")
    ps.append(p)

    cs = e.get_make_resident_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev9", "Make Resident % Time")
    ps.append(p)
    
    cs = e.get_resident_pages_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev10", "Resident Pages % Time")
    ps.append(p)

    cs = e.get_pop_pages_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev11", "Populate Pages % Time")
    ps.append(p)

    cs = e.get_unmap_res_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev12", "Unmap Mapping Range % Time", poly=False)
    ps.append(p)
    
    cs = e.get_unmap_rdup_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev13", "Unmap Rdup % Time")
    ps.append(p)

    cs = e.get_rmask_get_allocs()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev14", "RMask Get Alloc % Time")
    ps.append(p)

    cs = e.get_transfers() 
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-size-trans", "Unique Transfers")
    ps.append(p)

    #xs1, ys1, cs = zip(*[(x, y, c) for x, y, c in zip(xs, ys, cs) if y < 1.5e6])
    #p = size_time_plot_color(xs1, ys1, cs, psize, args, "transfer-sizev15", "GPU State % Time")
    cs = e.get_block_gpu_state_times()
    cs = [100 * c / y for c, y in zip(cs, ys)]
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev15", "GPU State % Time")
    ps.append(p)

    #cs = e.get_map_cpu_page_times()
    #p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev16", "Map CPU Page Time")
    #ps.append(p)
    
    #cs = e.get_pmm_add_map_times()
    #p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev17", "PMM Add Map")
    #ps.append(p)

    cs = e.get_num_new_vablocks()
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev18", "Num New VABlocks")
    ps.append(p)

    cs = e.get_unmap_res_counts()
    p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-sizev19", "Unmapped Resident Pages")
    ps.append(p)


    cs = e.get_prefetches_size()
    if len(cs) > 0:
    #xs1, ys1, cs = zip(*[(x, y, c) for x, y, c in zip(xs, ys, cs) if y < 1.5e6])
    #p = size_time_plot_color(xs1, ys1, cs, psize, args, "transfer-sizev15", "GPU State % Time")
        cs = e.num_true_prefetches()
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-size-pref", "# Prefetches")
        ps.append(p)

    evicts = e.get_evict_ops()
    #evicts = e.get_evicts()
    if sum(evicts) > 0:
        xs = e.get_transfer_size()
        cs = evicts
        p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-size-evict", "Evictions", legend=False, poly=False)
        #p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-size-evict", "Evictions", cmap="plasma", legend=False, poly=False)
        ps.append(p)
        #cs = e.get_evict_ops()
        #p = size_time_plot_color(xs, ys, cs, psize, args, "transfer-size-evict-ops", "Eviction Ops")
        #ps.append(p)



    if False:
        xs = e.get_0_transfer_batches()
        if len(xs) > 0:
            ys = e.get_0_transfer_times() 

            print (args.csv, "no transfers batch %:", 100 * len(xs) / len(e.get_total_faults()))
            print (args.csv, "no transfers time %:", 100 * sum(ys) / sum(e.get_batch_times()))
            size_time_plot(xs, ys, psize, args, "no-transfers")

            xs = e.get_0_transfer_batches_coal()
            ys = e.get_0_transfer_times() 
            size_time_plot(xs, ys, psize, args, "no-transfers-coal")

            xs = e.get_0_maps()
            ys = e.get_0_transfer_times() 
            cs = e.get_0_transfer_batches()
            p = size_time_plot_color(xs, ys, cs, psize, args, "maps", "Batch Data Migration Size (KB)")
            ps.append(p)

            xs = e.get_0_map_ops()
            ys = e.get_0_transfer_times() 
            cs = e.get_0_transfer_batches()
            p = size_time_plot_color(xs, ys, cs, psize, args, "map-ops", "Batch Data Migration Size (KB)")
            ps.append(p)
        else:
            print ("no 0 transfer batches")

    print ("waiting on plots")
    for p in ps:
        p.join()


if __name__== "__main__":
  main()

