import numpy as np
from scipy.interpolate import interp1d
import networkx as nx
import pyflagser
import typing as T

import path_homology


def vectorize_curve(
    filtration_levels: np.ndarray, curve: np.ndarray, counts_num: int = 256
) -> np.ndarray:
    curve_f = interp1d(
        filtration_levels,
        curve,
        kind='previous',
        bounds_error=False,
        fill_value=(0, curve[-1])
    )
    return np.array([curve_f(t) for t in np.linspace(0., 1., counts_num)])


def compute_directed_simplicial_betti_curves(
    w_adj_m: np.ndarray,
    sublevels_edges: T.Dict[float, T.Tuple[np.ndarray, np.ndarray]]
) -> T.Dict[str, T.Union[np.ndarray, T.Tuple[np.ndarray]]]:
    adj_m_t = np.full(w_adj_m.shape, False)
    betti_numbers_list = []
    euler_numbers_list = []
    max_dim = 0

    for t in sublevels_edges.keys():
        adj_m_t[sublevels_edges[t]] = True
        flagser_out = pyflagser.flagser_unweighted(adj_m_t)
        cur_max_dim = len(flagser_out['betti']) - 1
        if  cur_max_dim > max_dim:
            max_dim = cur_max_dim
        betti_numbers_list.append(flagser_out['betti'])
        euler_numbers_list.append(flagser_out['euler'])

    betti_numbers_list = [
        cur_session_betti_numbers \
        + (max_dim - len(cur_session_betti_numbers) + 1) * [0]
        for cur_session_betti_numbers in betti_numbers_list
    ]

    return {
        'betti_curves': tuple(np.array(betti_numbers)
                        for betti_numbers in zip(*betti_numbers_list)),
        'euler_curve': np.array(euler_numbers_list)
    }


def compute_path_betti_curves(
    w_adj_m: np.ndarray,
    sublevels_edges: T.Dict[float, T.Tuple[np.ndarray, np.ndarray]]
) -> T.Dict[str, T.Union[np.ndarray, T.Tuple[np.ndarray]]]:
    adj_m_t = np.full(w_adj_m.shape, False)
    betti_numbers_list = []
    euler_numbers_list = []
    max_dim = 0

    for t in sublevels_edges.keys():
        adj_m_t[sublevels_edges[t]] = True
        g = nx.DiGraph(adj_m_t)
        condensation_adj_m_t = nx.to_numpy_array(nx.condensation(g))
        cur_max_dim = 1
        if  cur_max_dim > max_dim:
            max_dim = cur_max_dim
        betti_numbers = [
            path_homology.utils.compute_path_homology_dimension(
                path_homology.graph.Graph(condensation_adj_m_t), dim
            ) for dim in range(cur_max_dim + 1)
        ]
        betti_numbers_list.append(betti_numbers)
        euler_numbers_list.append([-b if i % 2 else b for i, b in enumerate(betti_numbers)])

    betti_numbers_list = [
        cur_session_betti_numbers \
        + (max_dim - len(cur_session_betti_numbers) + 1) * [0]
        for cur_session_betti_numbers in betti_numbers_list
    ]

    return {
        'betti_curves': tuple(np.array(betti_numbers)
                        for betti_numbers in zip(*betti_numbers_list)),
        'euler_curve': np.array(euler_numbers_list)
    }


def compute_hochschild_betti_curves(
    w_adj_m: np.ndarray,
    sublevels_edges: T.Dict[float, T.Tuple[np.ndarray, np.ndarray]]
) -> T.Dict[str, T.Union[np.ndarray, T.Tuple[np.ndarray]]]:
    betti_numbers_list = []
    euler_numbers_list = []

    adj_m_t = np.full(w_adj_m.shape, False)
    g = nx.DiGraph()
    g.add_nodes_from(range(adj_m_t.shape[0]))

    for t in sublevels_edges.keys():
        adj_m_t[sublevels_edges[t]] = True
        g.add_edges_from([(u, v) for u, v in zip(*np.where(adj_m_t))])
        g = nx.condensation(g)
        paths_num = {
            e: sum(1 for _ in nx.all_simple_paths(g, e[0], e[1]))
            for e in g.edges()
        }

        dim_HH_0 = nx.number_weakly_connected_components(g)
        dim_HH_1 = dim_HH_0 - g.number_of_nodes() + sum(paths_num.values())
        betti_numbers_list.append([dim_HH_0, dim_HH_1])
        euler_numbers_list.append(dim_HH_0 - dim_HH_1)

    return {
        'betti_curves': tuple(np.array(betti_numbers)
                        for betti_numbers in zip(*betti_numbers_list)),
        'euler_curve': np.array(euler_numbers_list)
    }


def compute_dowker_betti_curves(w_adj_mat: np.ndarray):
    pass


def compute_betti_curves(g: nx.DiGraph, homology_types: T.List[str]):
    w_adj_m = nx.adjacency_matrix(g).toarray()
    filtration_levels = np.sort(np.unique(w_adj_m[w_adj_m != 0.]))
    sublevels_edges = {t: np.where(w_adj_m <= t) for t in filtration_levels}
    betti_curves = {
        homology_type: homology_betti_curves_compute_funcs[homology_type](w_adj_m, sublevels_edges)
        for homology_type in homology_types
    }
    return filtration_levels, betti_curves


homology_betti_curves_compute_funcs = {
    'directed_simplicial': compute_directed_simplicial_betti_curves,
    'path': compute_path_betti_curves,
    'hochschild': compute_hochschild_betti_curves,
    'dowker': compute_dowker_betti_curves
}


if __name__ == '__main__':
    adj_m = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0]
        ]
    )
    ph_out = path_homology.compute_path_homology_dimension(
        path_homology.graph.Graph(adj_m), 5
    )
    print(ph_out)
