import numpy as np

def pagerank(M, d=0.85, num_iterations=100):
# # experiemnt 2
# def pagerank(M, d=0.5, num_iterations=100):
    n = M.shape[0]
    E = np.ones(n) / n  # Initial uniform distribution vector
    # experiment 3
    # E = np.array([0.4, 0.3, 0.2, 0.1])
    R = E.copy()

    for _ in range(num_iterations):
        R_new = (1 - d) * E + d * M @ R
        if np.allclose(R, R_new):
            break
        R = R_new

    return R

# Example web graph (adjacency matrix)
# Rows represent outgoing links, columns represent incoming links
web_graph = np.array([[0, 1/2, 0, 0],
[1/3, 0, 0, 1/2],
[1/3, 0, 1, 1/2],
 [1/3, 1/2, 0, 0]])

# # experiment 4
# web_graph = np.array([[0, 1/5, 0, 0, 1/4, 1/6, 1/7],
#                     [1/3, 0, 0, 1/2, 1/4, 1/6, 1/7],
#                     [1/3, 0, 1, 1/2, 1/4, 1/6, 1/7],
#                     [1/3, 1/5, 0, 0, 1/4, 1/6, 1/7],
#                     [0, 1/5, 0, 0, 0, 1/6, 1/7],
#                     [0, 2/5, 0, 0, 0, 1/6, 1/7],
#                     [0, 0, 0, 0, 0, 0, 1/7]])

# # experiment 5
# size = 1000  
# shape = ((size, size))
# web_graph = np.random.rand(*shape)


# Normalize columns
M = web_graph / web_graph.sum(axis=0)

# Calculate PageRank scores
pagerank_scores = pagerank(M)

print("PageRank scores:")
for score in pagerank_scores:
    print(f"{score:.4f},")
