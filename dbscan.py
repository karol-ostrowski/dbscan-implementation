import sys
from pathlib import Path
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import time

class Point():
    def __init__(self, x, y, dist):
        self._was_visited = False
        self._cluster = 0
        self._coords = (float(x), float(y))
        self._neighbors = set()
        self._dist_to_ref = dist

    @property
    def dist_to_ref(self):
        return self._dist_to_ref

    @dist_to_ref.setter
    def dist_to_ref(self, dist):
        self._dist_to_ref = dist

    @property
    def was_visited(self):
        return self._was_visited
    
    @was_visited.setter
    def was_visited(self, value):
        self._was_visited = value
    
    @property
    def cluster(self):
        return self._cluster

    @cluster.setter
    def cluster(self, value):
        self._cluster = value

    @property
    def coords(self):
        return self._coords

    @property
    def neighbors(self):
        return self._neighbors

    def add_neighbor(self, point):
        self._neighbors.add(point)

    def __repr__(self):
        if self.was_visited == 0:
            state = "not visited"
        else:
            state = "visited"
        return f"(x, y) = {self.coords}, {state}, belongs to cluster {self.cluster}"

def distance(x, y):
    return pow(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2), 0.5)

def load_data(input_data):
    path = Path.cwd() / input_data
    with open(path, "r") as f:
        content = f.read().split("\n")
    
    content = content[:-1]

    data_as_set = set()
    data_as_list = list()
    ref_point = (-2, -2)
    for line in content:
        coords = line.split(",")
        data_as_list.append(Point(coords[0], coords[1],
                            distance((float(coords[0]), float(coords[1])), ref_point)))
    
    data_as_set = set(data_as_list.copy())
    return data_as_set, data_as_list

def dbscan_bruteforce(dataset, min_pts, eps):
    region_query_bruteforce(dataset=dataset, eps=eps)
    cluster_id = 0
    for current_point in dataset:
        if current_point.was_visited:
            continue
        
        current_point.was_visited = True

        if len(current_point.neighbors) < min_pts:
            #cluster -1 stores noise points
            current_point.cluster = -1
        else:
            cluster_id += 1
            cluster_points(cluster_id=cluster_id,
                           point=current_point,
                           min_pts=min_pts)

def region_query_bruteforce(dataset, eps):
    for p in dataset:
        for q in dataset:
            dist = distance(p.coords, q.coords)
            if dist < eps and dist > 0:
                p.add_neighbor(q)

def dbscan_triangle_inequality(dataset, min_pts, eps):
    dataset.sort(key=lambda point: point.dist_to_ref)
    region_query_triangle_inequality(dataset=dataset, eps=eps)
    cluster_id = 0
    for current_point in dataset:
        if current_point.was_visited:
            continue
        
        current_point.was_visited = True

        if len(current_point.neighbors) < min_pts:
            #cluster -1 stores noise points
            current_point.cluster = -1
        else:
            cluster_id += 1
            cluster_points(cluster_id=cluster_id,
                           point=current_point,
                           min_pts=min_pts)

def region_query_triangle_inequality(dataset, eps):
    for idx, p in enumerate(dataset):
        possible_neighbors = set()
        i = 1
        while True:
            try:
                q = dataset[idx + i]
                if q.dist_to_ref - p.dist_to_ref > eps:
                    break
                possible_neighbors.add(q)
                i += 1
            except IndexError:
                break
        j = 1
        while True:
            try:
                r = dataset[idx - j]
                if p.dist_to_ref - r.dist_to_ref > eps:
                    break
                possible_neighbors.add(r)
                j += 1
            except IndexError:
                break
        for neighbor in possible_neighbors:
            if distance(p.coords, neighbor.coords) < eps and p != neighbor:
                p.add_neighbor(neighbor)

def cluster_points(cluster_id, point, min_pts):
    point.cluster = cluster_id
    de = deque([p for p in point.neighbors])
    seen = {p for p in point.neighbors}
    while len(de) != 0:
        current_point = de.popleft()
        if current_point.was_visited == False:
            current_point.was_visited = True
            if len(current_point.neighbors) >= min_pts:
                de.extend([n for n in current_point.neighbors if n not in seen])
                seen.update(current_point.neighbors)

        if current_point.cluster == 0 or current_point.cluster == -1:
            current_point.cluster = cluster_id

def evaluate_result(dataset):
    assigned_points = defaultdict(list)
    for p in dataset:
        assigned_points[p.cluster].append(p)
    if -1 in assigned_points:
        del assigned_points[-1]
    centroids = dict()
    for cluster in assigned_points:
        centroids[cluster] = (sum([p.coords[0] for p in assigned_points[cluster]]) \
                              / len(assigned_points[cluster]),
                              sum([p.coords[1] for p in assigned_points[cluster]]) \
                              / len(assigned_points[cluster]))
        
    # compares distances between clusters to their internal scatter (how spread out they are)
    # measures compactness and seperation
    silhouette_score(dataset=dataset, centroids=centroids)

    # compares the distance to own cluster to the distance to next nearest cluster
    # measures quality of assignment
    davies_bouldin_index(centroids=centroids, assigned_points=assigned_points)

def davies_bouldin_index(centroids, assigned_points):
    num_of_clusters = len(centroids)
    cluster_scatter = dict()
    for id, coords in centroids.items():
        current_sum = 0
        for p in assigned_points[id]:
            current_sum += distance(p.coords, coords)
        cluster_scatter[id] = current_sum / len(assigned_points[id])
    
    max_ratio = float("-inf")
    max_ratios = list()
    for id1, _ in centroids.items():
        for id2, _ in centroids.items():
            if id1 == id2:
                continue
            if (
                (cluster_scatter[id1] + cluster_scatter[id2]) / 
                distance(centroids[id1], centroids[id2]) > max_ratio
                ):
                max_ratio = (cluster_scatter[id1] + cluster_scatter[id2]) / \
                            distance(centroids[id1], centroids[id2])
        max_ratios.append(max_ratio)
    db_index = sum(max_ratios) / num_of_clusters
    print("davies bouldin index:", db_index)

def silhouette_score(dataset, centroids):
    closest_clusters = dict()
    for id1 in centroids:
        if id1 != 1:
            closest_clusters[id1] = 1
        else:
            closest_clusters[id1] = 2
        for id2 in centroids:
            if (
                id1 != id2
                and distance(centroids[id1], centroids[closest_clusters[id1]]) >
                distance(centroids[id1], centroids[id2])
            ):
                closest_clusters[id1] = id2
    sum_of_silhouette_scores = 0
    num_of_noise_points = 0
    for p in dataset:
        if p.cluster == -1:
            num_of_noise_points += 1
            continue
        sum_infra_dist = 0
        infra_count = 0
        sum_inter_dist = 0
        inter_count = 0
        for q in dataset:
            if q.cluster == -1:
                continue
            elif p.cluster == q.cluster:
                sum_infra_dist += distance(p.coords, q.coords)
                infra_count += 1
            elif q.cluster == closest_clusters[p.cluster]:
                sum_inter_dist += distance(p.coords, q.coords)
                inter_count += 1
        try:
            avg_infra_dist = sum_infra_dist / infra_count
            avg_inter_dist = sum_inter_dist / inter_count
            sum_of_silhouette_scores += (avg_inter_dist - avg_infra_dist) \
                                        / max(avg_inter_dist, avg_infra_dist)
        except ZeroDivisionError:
            print("error: all points were assigned as noise")
    print("silhouette score:", sum_of_silhouette_scores / (len(dataset) - num_of_noise_points))


if __name__ == "__main__":
    input_dataset, eps, min_pts = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    data_as_set, data_as_list = load_data(input_data=input_dataset)

    bruteforce_time_start = time.time()
    dbscan_bruteforce(dataset=data_as_set, min_pts=min_pts, eps=eps)
    print("time taken for the bruteforce approach:", time.time() - bruteforce_time_start)

    TI_time_start = time.time()
    dbscan_triangle_inequality(dataset=data_as_list, min_pts=min_pts, eps=eps)
    print("time taken for the TI approach:", time.time() - TI_time_start)

    clusters = list(set(p.cluster for p in data_as_list))

    evaluate_result(dataset=data_as_list)

    colors = {cluster: plt.cm.tab20(i) for i, cluster in enumerate(clusters)}
    for cluster in clusters:
        group_points = [p.coords for p in data_as_list if p.cluster == cluster]
        xs, ys = zip(*group_points)
        plt.scatter(xs, ys, label=f"cluster {cluster}", color=colors[cluster])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("dbscan result")
    plt.legend()
    plt.grid(True)
    plt.show()