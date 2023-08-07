import argparse
import pandas as pd


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
            entries.append(batch_id)
            current_batch.append(entries[1:])  # Exclude the leading 'f'
        else:
            header = line.strip().split(',')
            headers.append(header)

    return headers, batches


def main(args):
    file_data = {}  # Dictionary to store headers and batches per file

    for file in args.files:
        headers, batches = parse_file(file)

        header_cols = ['base_address', 'allocation_length']
        headers_df = pd.DataFrame(headers, columns=header_cols)

        batch_cols = [
            "fault_address", "timestamp", "fault_type", "fault_access_type", "access_type_mask", "num_instances",
            "client_type", "mmu_engine_type",
            "client_id", "mmu_engine_id",
            "utlb_id", "gpc_id", "channel_id",
            "ve_id", "batch_id"
        ]
        batches_df = pd.concat([pd.DataFrame(batch, columns=batch_cols) for batch in batches], ignore_index=True)

        file_data[file] = {'headers': headers_df, 'batches': batches_df}

    # Here you can process the individual files' data stored in file_data
    for file, data in file_data.items():
        print(f"File: {file}")
        print("Headers:")
        print(data['headers'])
        print("Batches:")
        print(data['batches'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse text files containing memory access traces.')
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                        help='The text files to be parsed.')

    args = parser.parse_args()
    main(args)
