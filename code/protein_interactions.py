from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, SpectralEmbedding, Isomap, TSNE
import re
from os import path, makedirs

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

assert len(get_components(new_graph)) == 1, "more than 1 component"

eigvals, eigvects = np.linalg.eigh(dists)
print("Unique Eigenvalues:", len(eigvals) == len(set(eigvals)))
print(len(eigvals), "total vs.", len(set(eigvals)), "unique\n")  # might cause issue for spectral embedding?


# issue might simply be sparsity, which conflicts with the default "affinity" parameter of KNN

# %% Plotting
def plot_2d(data, title, save_to_dir: str = None, labels=None):
    assert data.shape[1] == 2, "Shouldn't do 2 component plot with more than 2 components"

    plt.clf()
    plt.figure()
    ax = sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels)
    ax.set_title(title)
    ax.set_xlabel('Component 1'), ax.set_ylabel('Component 2')

    if save_to_dir:
        plt.savefig(path.join(save_to_dir, re.sub("[()-]", "", title).replace(" ", "_").lower()))
        plt.show()
    else:
        plt.show()


def plot_3d(data, title, save_to_dir: str = None, labels=None):
    assert data.shape[1] == 3, "Shouldn't do 3 component plot with more than 2 components"

    plt.clf()
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    if save_to_dir:
        plt.savefig(path.join(save_to_dir, re.sub("[()-]", "", title).replace(" ", "_").lower()))
        plt.show()
    else:
        plt.show()


def jitter(data, magnitude: float):
    assert magnitude > 0.
    mask = np.random.uniform(-magnitude, magnitude, data.shape)
    return data + mask


# %% Manifold Embeddings + PCA

data = dists.to_numpy()
components = 3
save_to_dir = "../figures"  # None
if save_to_dir:
    makedirs(save_to_dir, exist_ok=True)

mds = MDS(n_components=components)
mds_embedding = mds.fit_transform(data)

se = SpectralEmbedding(n_components=components, affinity="rbf")
se_embedding = se.fit_transform(data)

iso = Isomap(n_components=components)
iso_embedding = iso.fit_transform(data)

tsne = TSNE(n_components=components)
tsne_embedding = tsne.fit_transform(data)

pca = PCA(n_components=components)
pca_reduction = pca.fit_transform(data)

if components == 2:
    plot = plot_2d
elif components == 3:
    dir(mplot3d)  # use the import once so it doesn't get "optimized" out
    plot = plot_3d
else:
    raise not NotImplementedError("Just not ready for that, bro")

# need to jitter because many points overlap
plot(jitter(mds_embedding, 0.5), f"MDS Embedding ({components}D)", save_to_dir=save_to_dir)
plot(jitter(iso_embedding, 0.5), f"ISOMAP Embedding ({components}D)", save_to_dir=save_to_dir)
plot(jitter(tsne_embedding, 0.1), f"t-SNE Embedding ({components}D)", save_to_dir=save_to_dir)
plot(jitter(se_embedding, 0.002), f"Spectral Embedding ({components}D)", save_to_dir=save_to_dir)
plot(jitter(pca_reduction, 0.5), f"PCA Reduction ({components}D)", save_to_dir=save_to_dir)
