implementation of the dbscan algorithm in python based on the original research paper
(Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996.
A density-based algorithm for discovering clusters in large spatial databases with noise.
In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD'96))

datasets shown in the project documentation come from
https://github.com/deric/clustering-benchmark

script tests the viability of the dbscan algorithm of a few distinct datasets.
each run compares the time taken for completing the task using the naive approach and the triangle inequality approach for region querying.
after clustering is done the script scores the outcome using the silhouette score and the davis-bouldin index, then it displays plotted results.
slightly more detailed description can be found in the project documentation file.
