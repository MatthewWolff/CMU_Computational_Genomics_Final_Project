from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE

from graph import get_components, extract_component, floyd_warshall

# https://www.ebi.ac.uk/intact/, searched "ABL"
# as per https://www.proteinatlas.org/search/protein_class%3ACOSMIC+Somatic+Mutations)
data = pd.read_table("../data/data.txt")
pairs = pd.DataFrame(data[data.columns[:2]].to_numpy(), columns=["protein1", "protein2"])

# %% Exploration
print("total interactions in the dataset:", len(pairs.protein1))
print("total unique proteins in first column:", len(pairs.protein1.unique()))

proteins = set(pairs.protein1.unique()) | set(pairs.protein2.unique())
print("total proteins in dataset:", len(proteins))

print("Proteins with most interactions")
print(pairs.groupby("protein1").count().sort_values(by="protein2", ascending=False).head().reset_index())


# %% Constructing a graph, analyzing it, grabbing largest connected component

# "Although real-world PPI networks are generally non-fully connected,
#   they have one largest connected sub-network, which contains most of the networks, nodes and edges"

# "ISOMAP algorithm requires the analyzed manifold is a convex subset of RD.
#   Then, the dataset must be an open connected subset of RD (Donoho and Grimes, 2003).
#   Therefore, the ISOMAP algorithm can only handle fully connected PPI networks or the
#   largest connected component of the non-fully connected ones"

def make_graph(edges: pd.DataFrame) -> Dict:
    graph = defaultdict(set)
    for i, row in edges.iterrows():
        graph[row.protein1].add(row.protein2)
        graph[row.protein2].add(row.protein1)
    return graph


g = make_graph(pairs)
components = get_components(g)
largest_network_size = max(components.values())
node_from_largest = max(components.items(), key=lambda x: x[1])[0]
network = extract_component(g, node_from_component=node_from_largest)

print("disconnected components in graph:", len(components))
print(f"largest component: {largest_network_size} out of {len(data)} nodes "
      f"({round(largest_network_size / len(data) * 100, 3)}%)")
print("Example Network:")
print(*list(network.items())[:5], sep="\n", end="\n\n")


# %% Neighborhood construction & geodesic distance calculation

# "We introduce the weight matrix W for the graph where the elements are 1 if there's an edge, else 0"
def make_adjacency_matrix(graph):
    matrix = pd.DataFrame(index=graph.keys(), columns=graph.keys()).fillna(float("inf"))
    for k, values in graph.items():
        for v in values:
            matrix[k][v] = 1
        matrix[k][k] = 0  # set this afterwards --- each node is trivially its own neighbor, don't need loop
    return matrix


adj_matrix = make_adjacency_matrix(network)
dists = pd.DataFrame(floyd_warshall(adj_matrix.to_numpy()), index=network.keys(), columns=network.keys())
print("Distances:")
print(dists, "\n")

# %% check connected...
new_graph = dict()
for col in dists.columns:
    new_graph[col] = set(dists.index[dists[col] != 0])

assert len(get_components(new_graph)) == 1


# %% Plotting
def plot_2d(data, title, path: str = None, labels=None):
    assert data.shape[1] == 2, "Shouldn't do 2 component plot with more than 2 components"

    plt.clf()
    plt.figure()
    ax = sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels)
    ax.set_title(title)
    ax.set_xlabel('Component 1'), ax.set_ylabel('Component 2')

    if path is not None:
        plt.savefig(path)
        plt.show()
    else:
        plt.show()


def plot_3d(data, title, path: str = None, labels=None):
    assert data.shape[1] == 3, "Shouldn't do 3 component plot with more than 2 components"

    plt.clf()
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    if path is not None:
        plt.savefig(path)
        plt.show()
    else:
        plt.show()


# %% Manifold Embeddings + PCA

data = dists.to_numpy()
components = 3

mds = MDS(n_components=components)
mds_embedding = mds.fit_transform(data)

# se = SpectralEmbedding(n_components=components) # must be fully connected graph... or else error
# se_embedding = se.fit_transform(data)

iso = Isomap(n_components=components)
iso_embedding = iso.fit_transform(data)

tsne = TSNE(n_components=components)
tsne_embedding = tsne.fit_transform(data)

pca = PCA(n_components=components)
pca_reduction = pca.fit_transform(data)

if components == 2:
    plot_2d(mds_embedding, f"MDS Embedding ({components}D)")
    plot_2d(iso_embedding, f"ISOMAP Embedding ({components}D)")
    plot_2d(tsne_embedding, f"t-SNE Embedding ({components}D)")
    # plot_2d(se_embedding, f"Spectral Embedding ({components}D)")
    plot_2d(pca_reduction, f"PCA Reduction ({components}D)")
elif components == 3:
    dir(mplot3d)  # use the import once so it doesn't get "optimized" out
    plot_3d(mds_embedding, f"MDS Embedding ({components}D)")
    plot_3d(iso_embedding, f"ISOMAP Embeddinfg ({components}D)")
    plot_3d(tsne_embedding, f"t-SNE Embedding ({components}D)")
    # plot_3d(se_embedding, f"Spectral Embedding ({components}D)")
    plot_3d(pca_reduction, f"PCA Reduction ({components}D)")
