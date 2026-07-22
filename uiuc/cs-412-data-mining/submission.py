# Submit this file to Gradescope
from typing import List
import math
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.

class Solution:
  def hclus_single_link(self, X: List[List[float]], K: int) -> List[int]:
    """Single link hierarchical clustering"""
    clusters = [[i] for i in range(len(X))]

    while len(clusters) > K:
      min_dist = float("inf")
      pair = (0, 1)

      for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
          distances = [
            math.sqrt(sum((X[p][d] - X[q][d]) ** 2 for d in range(len(X[p]))))
            for p in clusters[i] for q in clusters[j]
          ]
          d = min(distances)
          if d < min_dist:
            min_dist = d
            pair = (i, j)

      i, j = pair
      new_cluster = clusters[i] + clusters[j]
      clusters = [clusters[k] for k in range(len(clusters)) if k not in (i, j)]
      clusters.append(new_cluster)

    labels = [0] * len(X)
    for cid, members in enumerate(clusters):
      for idx in members:
        labels[idx] = cid
    return labels

  def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
    """Complete link hierarchical clustering"""
    clusters = [[i] for i in range(len(X))]

    while len(clusters) > K:
        min_dist = float("inf")
        pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distances = [
                    math.sqrt(sum((X[p][d] - X[q][d]) ** 2 for d in range(len(X[p]))))
                    for p in clusters[i] for q in clusters[j]
                ]
                d = sum(distances) / len(distances)
                if d < min_dist:
                    min_dist = d
                    pair = (i, j)

        i, j = pair
        new_cluster = clusters[i] + clusters[j]
        clusters = [clusters[k] for k in range(len(clusters)) if k not in (i, j)]
        clusters.append(new_cluster)

    labels = [0] * len(X)
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels

  def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
    """Average link hierarchical clustering"""
    clusters = [[i] for i in range(len(X))]

    while len(clusters) > K:
        min_dist = float("inf")
        pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distances = [
                    math.sqrt(sum((X[p][d] - X[q][d]) ** 2 for d in range(len(X[p]))))
                    for p in clusters[i] for q in clusters[j]
                ]
                d = max(distances)
                if d < min_dist:
                    min_dist = d
                    pair = (i, j)

        i, j = pair
        new_cluster = clusters[i] + clusters[j]
        clusters = [clusters[k] for k in range(len(clusters)) if k not in (i, j)]
        clusters.append(new_cluster)

    labels = [0] * len(X)
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels
