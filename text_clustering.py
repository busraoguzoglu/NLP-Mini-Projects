# Text clustering using Levenshtein Distance

import numpy as np
from sklearn.cluster import AffinityPropagation
import distance
import pandas as pd

def main():

    # Specify filename
    filename = ''

    # Data specific preprocessing
    data = pd.read_csv(filename, header=None)
    data = data[1].values

    queries = []
    for i in range(300):
        queries.append(data[i])
    queries = np.asarray(queries)

    # 1. Affinity Propagation
    # Note that we cannot specify number of clusters beforehand
    lev_distance = -1 * np.array([[distance.levenshtein(w1, w2) for w1 in queries] for w2 in queries])
    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, max_iter=400, convergence_iter=50)
    affprop.fit(lev_distance)

    print('Number of clusters are:', len((np.unique(affprop.labels_))))

    for cluster_id in np.unique(affprop.labels_):
        exemplar = queries[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(queries[np.nonzero(affprop.labels_ == cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - **%s:** %s" % (exemplar, cluster_str))

if __name__ == '__main__':
    main()