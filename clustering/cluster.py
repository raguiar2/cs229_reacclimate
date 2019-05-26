import numpy as np
import csv
import os
from sklearn.cluster import KMeans

OUTPUT_CSV = 'clusters.csv'

def get_users_to_data(csvs):
    users_to_data = {}
    for csv_file in csvs:
        with open(csv_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                if i ==0:
                    screen_name_idx = line.index('screen_name')
                    polarity_idx = line.index('polarity')
                    subjectivity_idx = line.index('subjectivity')
                    location_idx = line.index('location')
                if i!=0:
                    screen_name = line[screen_name_idx]
                    polarity = line[polarity_idx]
                    subjectivity = line[subjectivity_idx]
                    # TODO: add this later
                    #location = line[location_idx]
                    if screen_name in users_to_data:
                        old_polarity, old_subjectivity, lat, lon = users_to_data[screen_name]
                        # running avg of polarity and subjectivity
                        polarity = (float(old_polarity) + float(polarity))/2
                        subjectivity = (float(old_subjectivity) + float(subjectivity)) / 2
                    users_to_data[screen_name] = np.array([float(polarity), float(subjectivity), 0, 0])
    return users_to_data

def cluster_users(user_tuples):
    kmeans = KMeans(n_clusters=8, random_state=0).fit(user_tuples)
    return kmeans

def main():
    data_csvs = ['../data/' + o for o in os.listdir('../data') if o[-4:] == '.csv']
    # average sentiment and location for clustering
    users_to_vectors = get_users_to_data(data_csvs)
    user_tuples = [(key, value) for key, value in users_to_vectors.items()]
    user_values = [tup[1] for tup in user_tuples]
    user_clusters = cluster_users(user_values).labels_
    output_mapping = []
    for user_tuple, cluster in zip(user_tuples, user_clusters):
        user_screen_name = user_tuple[0]
        output_mapping.append([user_screen_name, cluster])
    with open(OUTPUT_CSV, "w") as f:
        writer = csv.writer(f)
        writer.writerows(output_mapping)


if __name__ == '__main__':
    main()
