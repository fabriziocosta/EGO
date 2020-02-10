from ego.vectorize import set_feature_size, vectorize
from ego.encode import make_encoder
#from ego.decompose import concatenate, concatenate_disjunctive, compose, iterate, head_compose, abstract_compose

from ego.decompose import compose, args, do_decompose

# order 0
from ego.decomposition.identity import decompose_identity
from ego.decomposition.nodes_edges import decompose_nodes_and_edges, decompose_nodes, decompose_edges
from ego.decomposition.path import decompose_path
from ego.decomposition.paired_neighborhoods import decompose_paired_neighborhoods, decompose_neighborhood
from ego.decomposition.cycle_basis import decompose_cycles_and_non_cycles, decompose_non_cycles, decompose_cycles
from ego.decomposition.clique import decompose_clique_and_non_clique, decompose_clique, decompose_non_clique
from ego.decomposition.graphlet import decompose_graphlet
#from ego.decomposition.communities import decompose_communities
# node-edge filter based
from ego.decomposition.degree import decompose_degree_and_non_degree, decompose_degree, decompose_non_degree
from ego.decomposition.centrality import decompose_central_and_non_central, decompose_central, decompose_non_central
from ego.decomposition.positive_and_negative import decompose_positive, decompose_negative, decompose_positive_and_negative

# order 1
from ego.decomposition.size import decompose_node_size, decompose_edge_size 
from ego.decomposition.context import decompose_context
from ego.decomposition.dilatate import decompose_dilatate
#from ego.decomposition.union import decompose_union
from ego.decomposition.join import decompose_node_join, decompose_edge_join
from ego.decomposition.pair import decompose_pair

from ego.decomposition.iterated_clique import decompose_iterated_clique
from ego.abstraction.abstract import decompose_abstract, decompose_abstract_and_non_abstract

# order 2
from ego.decomposition.set import decompose_difference, decompose_symmetric_difference, decompose_union, decompose_intersection
from ego.decomposition.relation import decompose_relation
from ego.decomposition.pair_binary import decompose_pair_binary

# order 3
from ego.decomposition.relation_binary import decompose_relation_binary

# order n
from ego.decomposition.concatenate import decompose_concatenate

# preprocessors
from ego.abstraction.abstract_label import preprocess_abstract_label
from ego.abstraction.minor_graph import preprocess_minor_degree

from ego.learn import PartImportanceEstimator