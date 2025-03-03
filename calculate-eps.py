import sys
import matplotlib.pyplot as plt
from dbscan import distance, load_data

data, k = sys.argv[1], int(sys.argv[2])

_, data_as_list = load_data(data)
kth_distances = []

for p in data_as_list:
    distances = []
    for q in data_as_list:
        distances.append(distance(p.coords, q.coords))
    kth_distances.append(sorted(distances)[k])

kth_distances.sort()
plt.scatter(range(len(kth_distances)), kth_distances)
plt.grid(True)
plt.xlabel("index")
plt.ylabel("distance")
plt.title("distances to k-th nearest neighbor")
plt.show()