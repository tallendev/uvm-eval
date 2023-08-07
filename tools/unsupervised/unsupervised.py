import pandas as pd
import glob
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def parse_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    headers = []
    batches = []

    current_batch = []
    batch_id = 0

    for line in lines:
        if ',' not in line:
            continue
        elif line[0] == 's':
            current_batch = []
        elif line[0] == 'b':
            batches.append(current_batch)
            batch_id += 1
        elif line[0] == 'f':
            entries = line.strip().split(',')
            entries.append("batch_id")
            current_batch.append(entries[1:])  # Exclude the leading 'f'
        else:
            header = line.strip().split(',')
            headers.append(header)

    return headers, batches


def main(args):
    all_headers = []
    all_batches = []

    for file in args.files:
        headers, batches = parse_file(file)
        all_headers.extend(headers)
        all_batches.extend(batches)

    header_cols = ['base_address', 'allocation_length']
    headers_df = pd.DataFrame(all_headers, columns=header_cols)

    print(all_batches[0])
    batch_cols = [
        "fault_address", "timestamp", "fault_type", "fault_access_type", "access_type_mask", "num_instances",
        "client_type", "mmu_engine_type",
        "client_id", "mmu_engine_id",
        "utlb_id", "gpc_id", "channel_id",
        "ve_id", "batch_id"
    ]
    batches_df = pd.concat([pd.DataFrame(batch, columns=batch_cols) for batch in all_batches], ignore_index=True)

    batches_df['num_instances'] = batches_df['num_instances'].astype(int)
    batches_df['num_duplicate_faults_per_batch'] = batches_df.duplicated(subset=['batch_id', 'fault_address']).astype(int)

    features = batches_df[['num_instances', 'num_duplicate_faults_per_batch']]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(features_scaled)
    batches_df['cluster'] = kmeans.labels_

    plt.figure(figsize=(10, 7))
    plt.scatter(batches_df['num_instances'], batches_df['num_duplicate_faults_per_batch'], c=batches_df['cluster'])
    plt.xlabel('Number of Instances')
    plt.ylabel('Number of Duplicate Faults per Batch')
    plt.title('Clusters of Duplicate Faults')

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='r', zorder=10)

    plt.savefig("clusters.png", dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse text files containing memory access traces.')
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                        help='The text files to be parsed.')

    args = parser.parse_args()
    main(args)
