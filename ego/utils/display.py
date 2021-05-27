#!/usr/bin/env python

from forceatlas2 import forceatlas2_networkx_layout
from scipy.spatial import Voronoi
from scipy.spatial import Delaunay
from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from eden.display import draw_graph


def plot_pdf2D(mtx, n_components=10, n_levels=5):
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(mtx)

    data_bounds_mins, data_bounds_maxs = np.min(mtx, axis=0), np.max(mtx, axis=0)

    x = np.linspace(data_bounds_mins[0], data_bounds_maxs[0])
    y = np.linspace(data_bounds_mins[1], data_bounds_maxs[1])
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    logprob = gmm.score_samples(XX)
    Z = np.exp(logprob)
    minz, maxz = min(Z), max(Z)
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, Z, levels=np.linspace(minz, maxz, n_levels), alpha=.5, cmap='bone_r')


def get_canonical_orientation(pos):
    x = np.array([list(pos[i]) for i in pos])
    u, s, vh = np.linalg.svd(x, full_matrices=True)
    x = np.dot(x, vh.T)
    x = x.reshape(-1, 2)
    pos = {i: v for i, v in zip(pos, x)}
    return pos


def draw_bundled_graph(
        lattice,
        pos=None,
        n_iter=1,
        n_voronoi_iter=2,
        display_size=10,
        display_nodes=True,
        node_size=50,
        display_node_density=True,
        node_colormap='bwr',
        edge_colormap='cividis_r',
        edge_intensity=True):
    if pos is None:
        pos = forceatlas2_networkx_layout(
            lattice,
            niter=2000,
            outboundAttractionDistribution=False,  # Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
            edgeWeightInfluence=2.0,
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=False,
            barnesHutTheta=1.2,
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=0.01)
        pos = get_canonical_orientation(pos)

    points = np.array([pos[u] for u in pos])
    N = len(points)
    for iter in range(n_voronoi_iter):
        vor = Voronoi(points, qhull_options='Qbb Qc Qx')
        points = np.vstack([points, vor.vertices])
    vor = Voronoi(points, qhull_options='Qbb Qc Qx')
    M = len(vor.vertices)

    id2hash = {i: u for i, u in enumerate(pos)}
    hash2id = {u: i for i, u in enumerate(pos)}

    points = np.array([pos[u] for u in pos])
    all_points = np.vstack([points, vor.vertices])
    all_pos = {i: x for i, x in enumerate(all_points)}
    tri = Delaunay(all_points)

    G = nx.Graph()
    G.add_nodes_from(range(N + M))
    color = np.zeros(N)
    for i in range(N):
        G.nodes[i]['label'] = 0
        G.nodes[i]['type'] = 'lattice'
        G.nodes[i]['pos'] = all_pos[i]
        u = id2hash[i]
        color[i] = lattice.nodes[u].get('num_type', 0)

    for j in range(N, N + M):
        G.nodes[j]['label'] = 1
        G.nodes[j]['type'] = 'waypoint'
        G.nodes[j]['pos'] = all_pos[j]
    for simplex in tri.simplices:
        for i in range(len(simplex) - 1):
            u = simplex[i]
            v = simplex[i + 1]
            dist = np.sqrt(np.sum(np.power(G.nodes[u]['pos'] - G.nodes[v]['pos'], 2)))
            G.add_edge(u, v, label=0, weight=dist, count=0)
        G.add_edge(simplex[0], simplex[-1], label=0, weight=dist, count=0)

    for iter in range(n_iter):
        for u, v in lattice.edges():
            i = hash2id[u]
            j = hash2id[v]
            try:
                path = nx.shortest_path(G, source=i, target=j, weight='weight')
                for k in range(len(path) - 1):
                    a = path[k]
                    b = path[k + 1]
                    G.edges[a, b]['label'] = 1
                    G.edges[a, b]['count'] += 1
            except Exception:
                pass
        # normalize thicknes
        max_count = max(G.edges[i, j]['count'] for i, j in G.edges())
        for i, j in G.edges():
            G.edges[i, j]['thickness'] = 4 * G.edges[i, j].get('count', 0) / float(max_count)

        # update weights
        for i, j in G.edges():
            G.edges[i, j]['weight'] = 1.0 / (1 + G.edges[i, j].get('count', 0))
            G.edges[i, j]['count'] = 0

    args = dict(pos=all_pos, size=None, vertex_size=0, vertex_label=None, edge_label=None,
                edge_width='thickness', edge_alpha=1, edge_color='thickness', edge_colormap=edge_colormap, colormap=node_colormap)
    plt.figure(figsize=(1.62 * display_size, display_size))
    if edge_intensity:
        draw_graph(G, **args)
    else:
        draw_graph(G, edge_fixed_color='k', **args)

    x, y = points.T
    if display_nodes:
        colors = [lattice.nodes[u]['label'] for u in lattice.nodes()]
        if len(set(colors)) == 1:
            colors = 'w'
        plt.scatter(x, y, s=node_size, c=colors, edgecolors='k', linewidths=0.5, cmap=node_colormap)
    if display_node_density:
        plot_pdf2D(points)

    dx = (max(x) - min(x)) / 20
    dy = (max(y) - min(y)) / 20
    plt.xlim(min(x) - dx, max(x) + dx)
    plt.ylim(min(y) - dy, max(y) + dy)
    plt.show()
    return pos
