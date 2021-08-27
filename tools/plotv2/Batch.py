#!/usr/bin/python3
import sys
import os
import argparse
import collections
from collections import OrderedDict
from os.path import basename, splitext, dirname
from statistics import stdev
import numpy as np
#from iteration_utilities import duplicates



class Experiment:
    # send in csv file stupidly labeled .txt
    def __init__(self, c):
        print("Building experiment:", c)
        batches = []
        csv = None
        with open(c, 'r') as csvf:
            csv = csvf.readlines()
        for bid, line in enumerate(csv):
            cols = line.split(',')
            try:
                batches.append(Batch(bid, *cols[1:]))
            except TypeError as e:
                print(f"Missing args? {sys.exc_info()[0]}")
                print(c)
                print(line)
                print(cols)
                print("#########")
                sys.exit(1)
        self.batches = batches
        self.pf = "pf" in c
        self.start_time = batches[0].start_time

        assert (batches[0].start_time == min(batches, key=lambda b: b.start_time).start_time)

    def len_batches(self):
        return len(self.batches)

    def get_0_transfer_batches(self):
        return [batch.cached_faults for batch in self.batches if batch.num_transfers == 0]

    def get_0_transfer_batches_coal(self):
        return [batch.actual_faults for batch in self.batches if batch.num_transfers == 0]
    
    def get_0_transfer_times(self):
        return [batch.time for batch in self.batches if batch.num_transfers == 0]

    def get_0_maps(self):
        return [batch.num_mapped_pages for batch in self.batches if batch.num_transfers == 0]
    
    def get_0_map_ops(self):
        return [batch.num_mapping_ops for batch in self.batches if batch.num_transfers == 0]





    def get_batch_relative_start_times(self):
        return [batch.start_time - self.start_time for batch in self.batches]
    
    def get_batch_relative_end_times(self):
        return [batch.start_time + batch.time - self.start_time for batch in self.batches]

    def get_batch_times(self):
        return [batch.time for batch in self.batches]

    def get_no_duplicates(self):
        return [batch.actual_faults for batch in self.batches]
    
    def get_duplicates(self):
        return [batch.duplicate_faults for batch in self.batches]

    def get_total_faults(self):
        return [batch.cached_faults for batch in self.batches]

    def get_transfers(self):
        return [batch.num_transfers for batch in self.batches]
    
    # now in KB
    def get_transfer_size(self):
        return [batch.transfer_size / 1000 for batch in self.batches]

    def get_prefetches_size(self):
        return [batch.num_prefetch for batch in self.batches]
    
    def get_num_vablocks(self):
        return [batch.num_vablock for batch in self.batches]

    # double check this val
    # Actual faults should just be unique faults, prior to any info about residency
    def get_trimmed_faults(self):
        return [batch.not_already_resident for batch in self.batches]

    def num_already_resident(self):
        return [batch.cached_faults - batch.not_already_resident for batch in self.batches]

    #FIXME idk if this is right
    def num_true_prefetches(self):
        return [abs(batch.cached_faults - batch.not_already_resident) for batch in self.batches]

    def get_num_mapping_ops(self):
        return [batch.num_mapping_ops for batch in self.batches]

    def get_num_mapped_pages(self):
        return [batch.num_mapped_pages for batch in self.batches]


    def get_maps(self):
        return [batch.num_mapped_pages for batch in self.batches]
    
    def get_map_ops(self):
        return [batch.num_mapping_ops for batch in self.batches]

    def get_evicts(self):
        return [batch.num_evict for batch in self.batches]
    
    def get_evict_ops(self):
        return [batch.num_evict_ops for batch in self.batches]

    def get_root_chunk_allocs(self):
        return [batch.num_root_chunk_allocs for batch in self.batches]

    def get_transfer_times(self):
        return [batch.transfer_time for batch in self.batches]

    def get_map_times(self):
        return [batch.map_time for batch in self.batches]

    def get_unmap_times(self):
        return [batch.unmap_time for batch in self.batches]
    
    def get_service_times(self):
        return [batch.service_time for batch in self.batches]
    
    def get_make_resident_times(self):
        return [batch.make_resident_time for batch in self.batches]
    
    def get_resident_pages_times(self):
        return [batch.resident_pages_time for batch in self.batches]
    
    def get_pop_pages_times(self):
        return [batch.pop_pages_time for batch in self.batches]
    
    def get_unmap_res_times(self):
        return [batch.unmap_res_time for batch in self.batches]

    def get_unmap_rdup_times(self):
        return [batch.unmap_rdup_time for batch in self.batches]
    
    def get_rmask_get_allocs(self):
        return [batch.rmask_get_alloc for batch in self.batches]

    def get_block_gpu_state_times(self):
        return [batch.block_gpu_state_time for batch in self.batches]
    
    def get_map_cpu_page_times(self):
        return [batch.map_cpu_page_time for batch in self.batches]
    
    def get_pmm_add_map_times(self):
        return [batch.pmm_add_map_time for batch in self.batches]
    
    def get_num_new_vablocks(self):
        return [batch.num_new_vablock for batch in self.batches]
    
    def get_unmap_res_counts(self):
        return [batch.unmap_res_count for batch in self.batches]

class Batch:
    def __init__(self, batch_id, start_time, time, cached_faults, coalesced_faults, duplicate_faults, actual_faults,\
                not_already_resident, num_vablock, num_prefetch, num_evict, num_evict_ops, num_transfers, transfer_size,\
                num_mapped_pages, num_mapping_ops, root_chunk_allocs, batch_transfer_time, map_time, unmap_time,\
                service_time, make_resident_time, resident_pages_time, pop_pages_time, unmap_res_time, unmap_rdup_time,\
                rmask_get_alloc, block_gpu_state_time, num_new_vablock, unmap_res_count, num_replays, status=0):
                #rmask_get_alloc, block_gpu_state_time, map_cpu_page_time, pmm_add_map_time, num_replays, status=0):
        self.start_time = (int(start_time))
        self.batch_id = int(batch_id)
        self.time = int(time)
        self.cached_faults = int(cached_faults)
        self.coalesced_faults = int(coalesced_faults)
        self.duplicate_faults = int(duplicate_faults)
        self.actual_faults = int(actual_faults)
        self.not_already_resident = int(not_already_resident)
        self.num_vablock = int(num_vablock)
        self.num_prefetch = int(num_prefetch)
        self.num_evict = int(num_evict)
        self.num_evict_ops = int(num_evict_ops)
        self.num_transfers = int(num_transfers)
        self.transfer_size = int(transfer_size)
        self.num_mapped_pages = int(num_mapped_pages)
        self.num_mapping_ops = int(num_mapping_ops)
        #self.num_vablock_ro = int(num_vablock_ro)
        #self.num_not_already_resident_ro = int(num_not_already_resident_ro)
        self.num_root_chunk_allocs = int(root_chunk_allocs)
        #self.num_already_zero = int(already_zero)
        #self.num_already_zero_after_alloc = int(already_zero_after_alloc)
        #self.num_already_resident_pop = int (already_resident_pop)
        #self.num_populate_pages_op = int(populate_pages_op)
        #self.num_populated_pages = int(populated_pages)
        self.transfer_time = int(batch_transfer_time)
        self.map_time = int(map_time)
        self.unmap_time = int(unmap_time)
        self.service_time = int (service_time)
        self.make_resident_time = int(make_resident_time)
        self.resident_pages_time = int(resident_pages_time)
        self.pop_pages_time = int(pop_pages_time)
        self.unmap_res_time = int(unmap_res_time)
        self.unmap_rdup_time = int(unmap_rdup_time)
        self.rmask_get_alloc = int(rmask_get_alloc)
        self.block_gpu_state_time = int(block_gpu_state_time)
        #self.map_cpu_page_time = int(map_cpu_page_time)
        #self.pmm_add_map_time = int(pmm_add_map_time)
        self.num_new_vablock = int(num_new_vablock)
        self.unmap_res_count = int(unmap_res_count)
        
        self.num_replays = int(num_replays)
        self.status = int(status)

        assert (self.status == 0)
        assert (self.num_replays == 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Full path to CSV file')
    args = parser.parse_args()
    c = args.csv
    e = Experiment(c)
    print("time:", sum(e.get_batch_times()))
    #print((e.get_unmap_res_times()))
    #e.print_info()
    #e.print_short_batches()
    
    #e.print_duplicate_faults_4k()
    
        
    

if __name__ == "__main__":
        main()
