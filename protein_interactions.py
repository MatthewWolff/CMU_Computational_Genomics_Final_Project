from collections import defaultdict

import pandas as pd

from graph import get_components, extract_component, floyd_warshall

# https://www.ebi.ac.uk/intact/, searched "ABL"
# as per https://www.proteinatlas.org/search/protein_class%3ACOSMIC+Somatic+Mutations)
data = pd.read_table("data.txt")
pairs = pd.DataFrame(data[data.columns[:2]].to_numpy(), columns=["protein1", "protein2"])

# %% Exploration
print("total interactions in the dataset:", len(pairs.protein1))
print("total unique proteins in first column:", len(pairs.protein1.unique()))

proteins = set(pairs.protein1.unique()) | set(pairs.protein2.unique())
print("total proteins in dataset:", len(proteins))

print("Proteins with most interactions")
print(pairs.groupby("protein1").count().sort_values(by="protein2", ascending=False).head().reset_index())

# %% Constructing a graph, analyzing it, grabbing largest connected component
graph = defaultdict(set)
for i, row in pairs.iterrows():
    graph[row.protein1].add(row.protein2)
    graph[row.protein2].add(row.protein1)

# "Although real-world PPI networks are generally non-fully connected,
# they have one largest connected sub-network, which contains most of the networks, nodes and edges"

components = get_components(graph)
print("disconnected components in graph:", len(components))
largest_network_size = max(components.values())
print(f"largest component: {largest_network_size} out of {len(data)} nodes "
      f"({round(largest_network_size / len(data) * 100, 3)}%)")

# "ISOMAP algorithm requires the analyzed manifold is a convex subset of RD.
# Then, the dataset must be an open connected subset of RD (Donoho and Grimes, 2003).
# Therefore, the ISOMAP algorithm can only handle fully connected PPI networks or the
# largest connected component of the non-fully connected ones"

node_from_largest = max(components.items(), key=lambda x: x[1])[0]
network = extract_component(graph, node_from_component=node_from_largest)
network

# %% Neighborhood construction

# "We introduce the weight matrix W for the graph where the elements are 1 if there's an edge, else 0"
adj_matrix = pd.DataFrame(index=network.keys(), columns=network.keys()).fillna(float("inf"))
for k, values in network.items():
    for v in values:
        adj_matrix[k][v] = 1
    adj_matrix[k][k] = 0  # set this afterwards --- each node is its own neighbor

# %% Geodesic distances
dists = pd.DataFrame(floyd_warshall(adj_matrix.to_numpy()), index=network.keys(), columns=network.keys())
dists
