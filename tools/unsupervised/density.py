import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count

def scatter_plot_by_allocation_time(batches_df, headers, output_file, timelimit):
    # Iterate through each unique allocation
    min_timestamp = batches_df['timestamp'].min()
    for allocation in batches_df['allocation'].unique():
        subset = batches_df[(batches_df['allocation'] == allocation) & (batches_df['timestamp'] < (min_timestamp + timelimit))].drop_duplicates(subset='fault_address', keep='first')

        smallest_page = subset['fault_address'].min() // 4096
        largest_page = subset['fault_address'].max() // 4096
        allocation_length = headers[allocation] // 4096

        page_range = largest_page - smallest_page + 1

        density1 = len(subset) / page_range

        density2 = (page_range) / allocation_length

        plt.scatter(density2, density1, marker='*', label=hex(allocation))

    # create a scatter plot
    plt.xlabel('density2')
    plt.ylabel('density1')
    plt.title('scatter plot by allocation')
    plt.legend(fancybox=True, framealpha=0.5)
    plt.tight_layout()
    print(f"Saving output file {output_file}")
    plt.savefig(output_file, format='pdf')
    plt.close()


def scatter_plot_by_allocation_batchcap_application(batches_df, headers, output_file, batchcap):
    # Iterate through each unique allocation
    for allocation in batches_df['allocation'].unique():
        subset = batches_df[(batches_df['allocation'] == allocation) & (batches_df['batch_id'] < batchcap)].drop_duplicates(subset='fault_address', keep='first')

        smallest_page = subset['fault_address'].min() // 4096
        largest_page = subset['fault_address'].max() // 4096
        allocation_length = headers[allocation] // 4096

        page_range = largest_page - smallest_page + 1

        density1 = len(subset) / page_range

        density2 = (page_range) / allocation_length

        plt.scatter(density2, density1, marker='*', label=hex(allocation))

    # create a scatter plot
    plt.xlabel('density2')
    plt.ylabel('density1')
    plt.title('scatter plot by allocation')
    plt.legend(fancybox=True, framealpha=0.5)
    plt.tight_layout()
    print(f"Saving output file {output_file}")
    plt.savefig(output_file, format='pdf')
    plt.close()


def scatter_plot_by_allocation_batchcap_by_allocation(batches_df, headers, output_file, batchcap):
    # Iterate through each unique allocation
    for allocation in batches_df['allocation'].unique():
        first_batch_id = batches_df[(batches_df['allocation'] == allocation)]['batch_id'].min()
        subset = batches_df[(batches_df['allocation'] == allocation)  & (batches_df['batch_id'] < (batchcap + first_batch_id))].drop_duplicates(subset='fault_address', keep='first')

        smallest_page = subset['fault_address'].min() // 4096
        largest_page = subset['fault_address'].max() // 4096
        allocation_length = headers[allocation] // 4096

        page_range = largest_page - smallest_page + 1

        density1 = len(subset) / page_range

        density2 = (page_range) / allocation_length

        plt.scatter(density2, density1, marker='*', label=hex(allocation))

    # create a scatter plot
    plt.xlabel('density2')
    plt.ylabel('density1')
    plt.title('scatter plot by allocation')
    plt.legend(fancybox=True, framealpha=0.5)
    plt.tight_layout()
    print(f"Saving output file {output_file}")
    plt.savefig(output_file, format='pdf')
    plt.close()


def scatter_plot_by_allocation_batchcap_progression_by_allocation(batches_df, headers, output_file, batchcap):
    allocations = batches_df['allocation'].unique()
    colormap = plt.cm.viridis

    # Different markers to cycle through
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    marker_cycle = iter(markers)

    # Precompute the batches
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap

    grouped_data = batches_df.groupby(['allocation', 'batch_group'])

    for idx, allocation in enumerate(allocations):
        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")

        density1_values = []
        density2_values = []
        colors = []

        batch_groups = list(grouped_data.groups.keys())
        num_batch_groups = sum(1 for key in batch_groups if key[0] == allocation)

        for group_idx, ((alloc, batch_group), group_data) in enumerate(grouped_data):
            if alloc != allocation:
                continue

            group_data = group_data.drop_duplicates(subset='fault_address', keep='first')

            smallest_page = group_data['fault_address'].min() // 4096
            largest_page = group_data['fault_address'].max() // 4096
            allocation_length = headers[allocation] // 4096

            page_range = largest_page - smallest_page + 1

            density1 = len(group_data) / page_range
            density1_values.append(density1)

            density2 = (page_range) / allocation_length
            density2_values.append(density2)

            # Color based on the batch group progression
            if num_batch_groups > 1:
                colors.append(colormap(group_idx / (num_batch_groups - 1)))
            else:
                colors.append(colormap(0))

        current_marker = next(marker_cycle)

        print(f"Calling scatter for allocation: {hex(allocation)}...")
        plt.scatter(density2_values, density1_values, marker=current_marker, c=colors, label=hex(allocation))

    # Cleanup and final touches
    del batches_df['first_batch_id']
    del batches_df['batch_group']

    # Setup the plot
    plt.xlabel('density2')
    plt.ylabel('density1')
    plt.title('scatter plot by allocation')
    plt.legend(fancybox=True, framealpha=0.5)
    plt.tight_layout()
    print(f"Saving output file {output_file}")
    plt.savefig(output_file, format='pdf')
    plt.close()


def scatter_plot_by_allocation_batchcap_progression_by_allocation_rolling(batches_df, headers, output_file, batchcap):
    allocations = batches_df['allocation'].unique()
    colormap = plt.cm.viridis

    # Different markers to cycle through
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    marker_cycle = iter(markers)

    batchcap = batchcap // 2

    # Precompute the batches
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap

    grouped_data = batches_df.groupby(['allocation', 'batch_group'])

    for idx, allocation in enumerate(allocations):
        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")

        density1_values = []
        density2_values = []
        colors = []

        batch_groups = list(grouped_data.groups.keys())
        num_batch_groups = sum(1 for key in batch_groups if key[0] == allocation)

        old_group_data = pd.DataFrame()
        for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
            if alloc != allocation or group_idx == (len(grouped_data) - 1):
                continue
            group_data = pd.concat([group_data_original, old_group_data]).drop_duplicates(subset='fault_address', keep='first')

            smallest_page = group_data['fault_address'].min() // 4096
            largest_page = group_data['fault_address'].max() // 4096
            allocation_length = headers[allocation] // 4096

            page_range = largest_page - smallest_page + 1

            density1 = len(group_data) / page_range
            density1_values.append(density1)

            density2 = (page_range) / allocation_length
            density2_values.append(density2)


            # Color based on the batch group progression
            if num_batch_groups > 1:
                colors.append(colormap(group_idx / (num_batch_groups - 1)))
            else:
                colors.append(colormap(0))

            old_group_data = group_data_original

        current_marker = next(marker_cycle)

        print(f"Calling scatter for allocation: {hex(allocation)}...")
        plt.scatter(density2_values, density1_values, marker=current_marker, c=colors, label=hex(allocation))
        # ... [your previous code]

        # 1. Compute the statistics
        mean_density1 = np.mean(density1_values)
        median_density1 = np.median(density1_values)
        std_density1 = np.std(density1_values)

        mean_density2 = np.mean(density2_values)
        median_density2 = np.median(density2_values)
        std_density2 = np.std(density2_values)

        # Density1 statistics
        plt.axhline(y=mean_density1, color='r', linestyle='-', label=f'Mean Density1: {mean_density1:.2f}')
        plt.axhline(y=median_density1, color='g', linestyle='--', label=f'Median Density1: {median_density1:.2f}')
        plt.axhline(y=mean_density1 + std_density1, color='b', linestyle='-.', label=f'1 Std Dev Density1')
        plt.axhline(y=mean_density1 - std_density1, color='b', linestyle='-.')

        # Density2 statistics
        plt.axvline(x=mean_density2, color='y', linestyle='-', label=f'Mean Density2: {mean_density2:.2f}')
        plt.axvline(x=median_density2, color='c', linestyle='--', label=f'Median Density2: {median_density2:.2f}')
        plt.axvline(x=mean_density2 + std_density2, color='m', linestyle='-.', label=f'1 Std Dev Density2')
        plt.axvline(x=mean_density2 - std_density2, color='m', linestyle='-.')


    # Cleanup and final touches
    del batches_df['first_batch_id']
    del batches_df['batch_group']

    # Setup the plot
    plt.xlabel('density2')
    plt.ylabel('density1')
    plt.title('scatter plot by allocation')
    plt.legend(fancybox=True, framealpha=0.5)
    plt.tight_layout()
    print(f"Saving output file {output_file}")
    plt.savefig(output_file, format='pdf')
    plt.close()


# /home/tnallen/dev/uvm-eval/benchmarks/cublas/log_32768/cublas_0.txt
def scatter_plot_by_allocation_batchcap_progression_by_allocation_rolling_timeseries(batches_df, headers, output_file, batchcap):
    allocations = batches_df['allocation'].unique()

    # List of distinct colors and markers
    colors = plt.cm.Dark2.colors
    #colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

    batchcap = batchcap // 2
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap
    grouped_data = batches_df.groupby(['allocation', 'batch_group'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for idx, allocation in enumerate(allocations):
        if idx >= len(colors) or idx >= len(markers):  # Check to ensure we don't run out of unique colors or markers
            print(
                f"Warning: Ran out of unique colors and markers for allocation {idx}. Add more colors or markers, or combine some allocations.")
            continue

        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")

        batch_groups = []
        density1_values = []
        density2_values = []


        old_group_data = pd.DataFrame()
        for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
            if alloc != allocation:
                continue
            group_data = pd.concat([group_data_original, old_group_data]).drop_duplicates(subset='fault_address', keep='first')

            smallest_page = group_data['fault_address'].min() // 4096
            largest_page = group_data['fault_address'].max() // 4096
            allocation_length = headers[allocation] // 4096

            page_range = largest_page - smallest_page + 1

            density1 = len(group_data) / page_range
            density1_values.append(density1)

            density2 = (page_range) / allocation_length
            density2_values.append(density2)

            batch_groups.append(batch_group)

            old_group_data = group_data_original

        current_color = colors[idx]
        current_marker = markers[idx]

        # Plotting density1 and density2
        ax1.scatter(batch_groups, density1_values, marker=current_marker, color=current_color,
                    label=hex(allocation))
        ax2.scatter(batch_groups, density2_values, marker=current_marker, color=current_color,
                    label=hex(allocation))

        ax1.axhline(np.mean(density1_values), color=current_color, linestyle='-', label=f"Mean {hex(allocation)}")
        ax1.axhline(np.median(density1_values), color=current_color, linestyle='--',
                    label=f"Median {hex(allocation)}")

        ax2.axhline(np.mean(density2_values), color=current_color, linestyle='-', label=f"Mean {hex(allocation)}")
        ax2.axhline(np.median(density2_values), color=current_color, linestyle='--',
                    label=f"Median {hex(allocation)}")

    # Statistics for Density1 and Density2
    for ax, density_values, density_name in [(ax1, density1_values, "Density1"),
                                             (ax2, density2_values, "Density2")]:
        ax.set_title(f'Time vs. {density_name}')
        ax.set_xlabel('Time (Batch Group)')
        ax.set_ylabel(density_name)
        ax.legend()

    # Cleanup and final touches
    del batches_df['first_batch_id']
    del batches_df['batch_group']

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()


def scatter_plot_by_allocation_batchcap_progression_by_allocation_rolling_timeseries_movingavg(batches_df, headers, output_file, batchcap):
    allocations = batches_df['allocation'].unique()

    # List of distinct colors and markers
    colors = plt.cm.Dark2.colors
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

    batchcap = batchcap // 2
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap
    grouped_data = batches_df.groupby(['allocation', 'batch_group'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for idx, allocation in enumerate(allocations):
        if idx >= len(colors) or idx >= len(
                markers):  # Check to ensure we don't run out of unique colors or markers
            print(
                f"Warning: Ran out of unique colors and markers for allocation {idx}. Add more colors or markers, or combine some allocations.")
            continue

        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")

        batch_groups = []
        density1_values = []
        density2_values = []

        batch_group_keys = list(grouped_data.groups.keys())

        old_group_data = pd.DataFrame()
        for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
            if alloc != allocation:
                continue

            group_data = pd.concat([group_data_original, old_group_data]).drop_duplicates(subset='fault_address', keep='first')

            smallest_page = group_data['fault_address'].min() // 4096
            largest_page = group_data['fault_address'].max() // 4096
            allocation_length = headers[allocation] // 4096

            page_range = largest_page - smallest_page + 1

            density1 = len(group_data) / page_range
            density1_values.append(density1)

            density2 = (page_range) / allocation_length
            density2_values.append(density2)

            batch_groups.append(batch_group)

            old_group_data = group_data_original

        current_color = colors[idx]
        current_marker = markers[idx]

        # Moving averages
        ma_density1 = pd.Series(density1_values).rolling(window=100).mean().tolist()
        ma_density2 = pd.Series(density2_values).rolling(window=100).mean().tolist()

        # Plotting density1 and density2
        ax1.scatter(batch_groups, density1_values, marker=current_marker, color=current_color,
                    label=f"Scatter {hex(allocation)}")
        ax2.scatter(batch_groups, density2_values, marker=current_marker, color=current_color,
                    label=f"Scatter {hex(allocation)}")

        ax1.plot(batch_groups, ma_density1, color=current_color, linestyle='-.', alpha=0.5,
                 label=f"Moving Avg {hex(allocation)}")
        ax2.plot(batch_groups, ma_density2, color=current_color, linestyle='-.', alpha=0.5,
                 label=f"Moving Avg {hex(allocation)}")


    for ax, density_name in [(ax1, "Density1"), (ax2, "Density2")]:
        ax.set_title(f'Time vs. {density_name}')
        ax.set_xlabel('Time (Batch Group)')
        ax.set_ylabel(density_name)
        ax.legend()

    # Cleanup and final touches
    del batches_df['first_batch_id']
    del batches_df['batch_group']

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()


def compute_duplication_ratio(group_data, method):
    # Ensure 'num_instances' is treated as an integer
    group_data['num_instances'] = group_data['num_instances'].astype(int)

    unique_fault_addresses = group_data['fault_address'].nunique()

    if method == 'unique_addresses':
        return unique_fault_addresses

    elif method == 'inter':
        fault_counts = group_data['fault_address'].value_counts()
        duplicates = fault_counts.sum() - 1
        return duplicates / unique_fault_addresses

    elif method == 'intra':
        # all faults
        total_instances = group_data['num_instances'].sum()
        # inter
        fault_counts = group_data['fault_address'].value_counts()
        inter_duplicates = fault_counts.sum() - 1
        # all - inter
        return (total_instances - inter_duplicates) / unique_fault_addresses

    elif method == 'all':
        total_instances = group_data['num_instances'].sum() - 1
        return (total_instances - unique_fault_addresses) / unique_fault_addresses

    else:
        raise ValueError("Invalid method provided.")

def scatter_plot_by_allocation_batchcap_progression_by_allocation_duplicates_rolling_timeseries_movingavg(batches_df,
                                                                                                            headers,
                                                                                                            output_file,
                                                                                                            batchcap):
    allocations = batches_df['allocation'].unique()
    colors = plt.cm.Dark2.colors
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

    batchcap = batchcap // 2
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap
    grouped_data = batches_df.groupby(['allocation', 'batch_group'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    methods = ['unique_addresses', 'inter', 'intra', 'all']
    for idx, allocation in enumerate(allocations):
        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")

        for method_idx, method in enumerate(methods):
            batch_groups = []
            duplication_ratios = []

            old_group_data = pd.DataFrame()
            for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
                if alloc != allocation:
                    continue

                group_data = pd.concat([group_data_original, old_group_data])

                duplication_ratio = compute_duplication_ratio(group_data, method)

                batch_groups.append(batch_group)
                duplication_ratios.append(duplication_ratio)

                old_group_data = group_data_original

            current_color = colors[idx % len(colors)]
            current_marker = markers[idx % len(markers)]

            axs[method_idx // 2, method_idx % 2].scatter(batch_groups, duplication_ratios, marker=current_marker,
                                                         color=current_color,
                                                         label=f"{method} {hex(allocation)}")

            ma_ratio = pd.Series(duplication_ratios).rolling(window=100).mean().tolist()
            axs[method_idx // 2, method_idx % 2].plot(batch_groups, ma_ratio, color=current_color, linestyle='-.',
                                                      alpha=0.5)

    titles = ['Unique Addresses', 'Inter-SM Duplication Ratio', 'Intra-SM Duplication Ratio',
              'Overall Duplication Ratio']
    for i, title in enumerate(titles):
        axs[i // 2, i % 2].set_title(title)
        axs[i // 2, i % 2].set_xlabel('Time (Batch Group)')
        axs[i // 2, i % 2].set_ylabel('Value')
        axs[i // 2, i % 2].legend()

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()


def parse_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    headers = {}
    batches = []

    current_batch = []
    batch_id = 0

    for line in lines:
        #TODO remove me
        if ',' not in line:
            continue
        elif line[0] == 's':
            current_batch = []
        elif line[0] == 'b':
            batches.append(current_batch)
            batch_id += 1
        elif line[0] == 'e' or line[0] == 'p':
            continue
        elif line[0] == 'f':
            entries = line.strip().split(',')
            entries.append(batch_id)
            addr = int(entries[1], 16)
            entries[1] = addr
            entries[2] = int(entries[2]) # timestamp
            for header in headers.keys():
                #greater than base address and less than range end
                if addr >= header and addr < header + headers[header]:
                    entries.append(header)
                    break
            else:
                print("Fault found with no matching allocation")
            current_batch.append(entries[1:])  # Exclude the leading 'f'
        else:
            header = line.strip().split(',')
            print(header)
            # base address, allocation length
            # base addresses are base 16 but lengths are base 10 in bytes
            headers[int(header[0], 16)] = int(header[1], 10)
    print(f"headers: {headers}")
    return headers, batches

def get_output_file_name(input_file, specialization="", dirname=""):
    # Extract application_name and problem_size from the input file path
    application_name = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
    problem_size = os.path.basename(os.path.dirname(input_file)).split("_")[1]

    # Define the output directory
    output_dir = os.path.join(f"../../figures/density/{dirname}")
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file name
    output_file = os.path.join(output_dir, f"{application_name}-{problem_size}-density-{specialization}.pdf")

    return os.path.abspath(output_file)


def print_allocation_counts(df):
    # Check if 'allocation' column is in the dataframe
    if 'allocation' not in df.columns:
        print("Column 'allocation' not found in the DataFrame.")
        return

    unique_allocs = df['allocation'].unique()
    print(f"{len(unique_allocs)}")
    # Iterate through the unique allocations and print the count of matching rows
    for allocation in unique_allocs:
        count = len(df[df['allocation'] == allocation])
        print(f"Allocation: {allocation}, Number of matching rows: {count}")




def main(args):
    output_dir = "../../figures/density/"
    os.makedirs(output_dir, exist_ok=True)
    num_processes = cpu_count() - 1

    tasks = []
    with Pool(num_processes) as pool:
        results = pool.map(parse_and_construct_df, args.files)
        for batches_df, headers, file in results:
            for batchcap in [20, 50, 100, 200]:
                tasks.append((scatter_plot_by_allocation_batchcap_progression_by_allocation_duplicates_rolling_timeseries_movingavg,
                             batches_df, headers, get_output_file_name(file, f"allocation-batchcap-progression-duplicates-rolling-timeseries-movingavg-{batchcap}",
                             "allocation-continuous-batchcap-duplicates-rolling-timeseries-movingavg"), batchcap))
                # tasks.extend([
                #     (scatter_plot_by_allocation_batchcap_application, batches_df, headers,
                #      get_output_file_name(file, f"global-batchcap-{batchcap}", "global-batchcap"), batchcap),
                #     (scatter_plot_by_allocation_batchcap_by_allocation, batches_df, headers,
                #      get_output_file_name(file, f"allocation-batchcap-{batchcap}", "allocation-batchcap"), batchcap),
                #     (scatter_plot_by_allocation_batchcap_progression_by_allocation, batches_df, headers,
                #      get_output_file_name(file, f"allocation-batchcap-progression-{batchcap}",
                #                           "allocation-continuous-batchcap"), batchcap),
                #     (scatter_plot_by_allocation_batchcap_progression_by_allocation_rolling, batches_df, headers,
                #      get_output_file_name(file, f"allocation-batchcap-progression-rolling-{batchcap}",
                #                           "allocation-continuous-batchcap-rolling"), batchcap),
                #     (scatter_plot_by_allocation_batchcap_progression_by_allocation_rolling_timeseries, batches_df, headers,
                #      get_output_file_name(file, f"allocation-batchcap-progression-rolling-timeseries-{batchcap}",
                #                           "allocation-continuous-batchcap-rolling-timeseries"), batchcap),
                #     (scatter_plot_by_allocation_batchcap_progression_by_allocation_rolling_timeseries_movingavg, batches_df,
                #      headers,
                #      get_output_file_name(file, f"allocation-batchcap-progression-rolling-timeseries-movingavg-{batchcap}",
                #                           "allocation-continuous-batchcap-rolling-timeseries-movingavg"), batchcap)
                # ])

            #for timelimit in [int(1e5), int(1e6), int(1e7), int(1e8)]:
                #tasks.append((scatter_plot_by_allocation_time, batches_df, headers,
                #              get_output_file_name(file, f"global-timelimit-{timelimit}", "timelimit"), timelimit))

    print(tasks)
    print(f"Creating pool with {num_processes} processors for {len(tasks)} tasks:")
    with Pool(num_processes) as pool:
        pool.starmap(execute_task, tasks)


def parse_and_construct_df(file):
    print(f"Parsing file: {file}")
    headers, batches = parse_file(file)  # Make sure parse_file is defined elsewhere in your code
    batch_cols = [
        "fault_address", "timestamp", "fault_type", "fault_access_type",
        "access_type_mask", "num_instances", "client_type", "mmu_engine_type",
        "client_id", "mmu_engine_id", "utlb_id", "gpc_id", "channel_id",
        "ve_id", "batch_id", "allocation"
    ]

    batches_df = pd.concat([pd.DataFrame(batch, columns=batch_cols, dtype=object) for batch in batches],
                           ignore_index=True)
    print_allocation_counts(batches_df)  # Ensure that print_allocation_counts is defined elsewhere in your code

    return (batches_df, headers, file)


def execute_task(func, batches_df, headers, output_name, limit):
    func(batches_df, headers, output_name, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse text files containing memory access traces.')
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                        help='The text files to be parsed.')
    args = parser.parse_args()
    main(args)
