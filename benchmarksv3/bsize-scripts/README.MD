# About These Scripts
Each script will generate full datasets from the paper, including:
- prefetching on/off
- doubling batch sizes 32->4096, and 6144 (max) 
- listed problem sizes

Output will be produced in the specific application's directory. For example, executing "cublas.sh" will create
a series of directories ../cublas/ with the following format:\
log_{prefetchingf_}?{problem_size}_bsize_{batch_size}\
Where pf_ only appears if prefetching is enabled during the execution. The directories will contain one file per iteration inside the script (1 by default):\
cublas_{i} - output from the system log with only syslogger data\
cublas_{i}.txt - loose CSV formatting of syslogger data\
dmesg_1 - raw dmesg of most-recent execution for debugging\

### Data Format
Each line is a batch:\
b,363795213599097,736221,32,9,29,3,3,1,0,0,0,3,12288,3,1,1,15790,35196,0,534863,493736,18104,315251,34254,290,124063,112010,1,1,1\
with the format:\
batch_id, start_time, time, cached_faults, coalesced_faults, duplicate_faults, actual_faults, not_already_resident, num_vablock, num_prefetch, num_evict, num_evict_ops, num_transfers, transfer_size,num_mapped_pages, num_mapping_ops, root_chunk_allocs, batch_transfer_time, map_time, unmap_time, service_time, make_resident_time, resident_pages_time, pop_pages_time, unmap_res_time, unmap_rdup_time, rmask_get_alloc, block_gpu_state_time, num_new_vablock, unmap_res_count, num_replays, status=0\


batch_id: id from 0-max_batch-1\
start_time: start time of batch(ns)\
time: length of batch(ns)\
cached_faults: number of overall faults\
coalesced_faults: faults after coalescing faults with same address from same uTLB \
duplicate_faults: duplicates that are same address, different uTLB\
actual_faults: faults with all duplicates removed\
not_already_resident: faults  that require data migration (vs faults where data is on GPU and just need to be mapped)\
num_vablock: number of VABlocks with at least one fault in the batch\
num_prefetch: number of faults that are artificially inserted by prefetching\
num_evict: pages evicted with migration\
num_evict_ops: va_blocks requiring eviction\
num_transfers: number of separate data transfers required\
transfer_size: total amount of data moved\
num_mapped_pages: number of unmapped_pages\
num_mapping_ops: number of mapping operations required (more than one page may be mapped in each op, generally one per VABlock)\
root_chunk_allocs: number of root chunk allocations required (calls into nvidia RM) \
batch_transfer_time: time spent doing user data transfer for all batches \
map_time: time spent mapping data\
unmap_time: time spent unmapping GPU pages \
service_time: total time in the "service" portion of the driver \
make_resident_time: time spent in the "make_resident" function\
resident_pages_time: time spent in "resident_pages" subcall of make_resident\
pop_pages_time: time spent populating new pages with 0 \
unmap_res_time: time spent unmapping resident pages (CPU unmap)\
unmap_rdup_time: time spent unmapping read_only_duplicate pages\
rmask_get_alloc: Small overhead for getting a kernel memory mask allocation\
block_gpu_state_time: time spent in the block_gpu_state time (DMA stuff)\
num_new_vablock: number of new VABlocks added to GPU\
unmap_res_count: number of unmapped resident pages\
num_replays: number of fault replays (generally 1 per batch)\
status: optional field that only occurs when there is a batch handling error (unlikely for succesful execution)
