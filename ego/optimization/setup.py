from ego.vectorize import set_feature_size, vectorize
from ego.encode import make_encoder
#from ego.decompose import concatenate, concatenate_disjunctive, compose, iterate, head_compose, abstract_compose

from ego.decompose import compose, args, do_decompose, add, com

# node relabeling
from ego.decomposition.node_relabel import decompose_nodes_relabel_degree, decompose_nodes_relabel_null, decompose_nodes_relabel_mapped
from ego.decomposition.node_relabel import ndsrlbdgr, ndsrlbnll, ndsrlbmpd

# order 0
from ego.decomposition.identity import decompose_identity
from ego.decomposition.identity import idn
from ego.decomposition.nodes_edges import decompose_nodes_and_edges, decompose_nodes, decompose_edges
from ego.decomposition.nodes_edges import nds, edg, ndsedg
from ego.decomposition.path import decompose_path
from ego.decomposition.path import pth
from ego.decomposition.paired_neighborhoods import decompose_paired_neighborhoods, decompose_neighborhood
from ego.decomposition.paired_neighborhoods import ngb, prdngb
from ego.decomposition.cycle_basis import decompose_cycles_and_non_cycles, decompose_non_cycles, decompose_cycles
from ego.decomposition.cycle_basis import cyc, cycn, ncyc
from ego.decomposition.clique import decompose_clique_and_non_clique, decompose_clique, decompose_non_clique
from ego.decomposition.clique import clq, nclq, clqn
from ego.decomposition.graphlet import decompose_graphlet
from ego.decomposition.graphlet import grp
from ego.decomposition.communities import decompose_communities
from ego.decomposition.communities import cmm
from ego.decomposition.dbreak import decompose_break
from ego.decomposition.dbreak import brk

# node-edge filter based
from ego.decomposition.degree import decompose_degree_and_non_degree, decompose_degree, decompose_non_degree
from ego.decomposition.degree import dgr, ndgr, dgrn
from ego.decomposition.centrality import decompose_central_and_non_central, decompose_central, decompose_non_central
from ego.decomposition.centrality import cnt, ncnt, cntn
from ego.decomposition.positive_and_negative import decompose_positive, decompose_negative, decompose_positive_and_negative
from ego.decomposition.positive_and_negative import pst, ngt, pstngt

# order 1
from ego.decomposition.size import decompose_node_size, decompose_edge_size 
from ego.decomposition.size import ndsz, edgsz
from ego.decomposition.context import decompose_context
from ego.decomposition.context import cntx
from ego.decomposition.dilatate import decompose_dilatate
from ego.decomposition.dilatate import dlt
from ego.decomposition.union import decompose_all_union
from ego.decomposition.union import allunn
from ego.decomposition.join import decompose_node_join, decompose_edge_join
from ego.decomposition.join import ndjn, edgjn 
from ego.decomposition.pair import decompose_pair
from ego.decomposition.pair import par
from ego.decomposition.frequency import decompose_frequency
from ego.decomposition.frequency import frq
from ego.decomposition.relabel import decompose_relabel_estimator, decompose_relabel_distinct_node_labels, decompose_relabel_max_node_degree, decompose_relabel_node_size, decompose_relabel_node_label_frequency, decompose_relabel_node_degree_frequency
from ego.decomposition.relabel import rlbest, rlbmdgr, rlbdfrq, rlbnod, rlblfrq, rlbsiz
from ego.decomposition.iterated_clique import decompose_iterated_clique
from ego.decomposition.iterated_clique import itrclq
from ego.abstraction.abstract import decompose_abstract, decompose_abstract_and_non_abstract
from ego.abstraction.abstract import abst, abstn

# order 2
from ego.decomposition.set import decompose_difference, decompose_symmetric_difference, decompose_union, decompose_intersection
from ego.decomposition.set import dff, symdff, unn, intr
from ego.decomposition.relation import decompose_relation
from ego.decomposition.relation import rlt
from ego.decomposition.pair_binary import decompose_pair_binary
from ego.decomposition.pair_binary import parbnr

# order 3
from ego.decomposition.relation_binary import decompose_relation_binary
from ego.decomposition.relation_binary import rltbnr

# order n
from ego.decomposition.concatenate import decompose_concatenate

# preprocessors
from ego.abstraction.abstract_label import preprocess_abstract_label
from ego.abstraction.minor_graph import preprocess_minor_degree

from ego.learn import PartImportanceEstimator