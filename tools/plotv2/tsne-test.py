import argparse
import csv
import numpy as np
from sklearn.manifold import TSNE

def load_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        batch = []
        i = 0
        for row in reader:
            if row[0] == 's':
                batch = []
                data.append(batch)
            elif row[0].startswith('f'):
                batch.append([x for x in row])
            print(f"batch {i}: {batch}")
    return [lst for lst in data if lst]

def process_data(data):
    X = []
    for batch in data:
        start = None
        for row in batch:
            if row[0] == 0:
                start = row[1:]
            else:
                x = np.array(row[1:])
                X.append(x)
        X.append(np.array(start))
    return np.vstack(X)

def tsne_transform(X):
    tsne = TSNE(n_components=2, random_state=0, verbose=1)
    print(X)
    print (tsne)
    return tsne.fit_transform(X)

def save_csv(X, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in X:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input CSV file")
    args = parser.parse_args()
    input_file = args.input_file
    data = load_csv(input_file)
    X = process_data(data)
    transformed_X = tsne_transform(X)
    save_csv(transformed_X, 'output.csv')

if __name__ == '__main__':
    main()
