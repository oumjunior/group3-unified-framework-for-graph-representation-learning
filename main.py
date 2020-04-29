import magicgraph
from gensim.models import Word2Vec
from numpy import savetxt

# To be able to run on a remote machine
import matplotlib
matplotlib.use('Agg')

import lib.HARP.src.graph_coarsening as graph_coarsening
from lib.node2vec_evaluations import *
from lib.struc2vec_evaluations import *
from lib.Harp_evaluation import *
from lib.walklets_evaluations import *
from lib.graph import read_graph
from lib.node2vec.node2vec import Graph
from lib.struc2vec import package_1
from lib.walklets.walklets import Walklets


def generate_node2vec_embeddings(input_file, p, q):
    """
    Generates node2vec embeddings

    Author: Stephen Banin Panyin
    """

    output_folder = "output/embeddings/"
    output_file = input_file.split("/")[-1].replace(".edgelist", ".node2vec")
    output_filename = output_folder + output_file

    # Reading Graph
    G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    G = G.to_undirected()
    G = Graph(G, False, 1, 1)
    G.preprocess_transition_probs()

    # Learning Embeddings
    walks = G.simulate_walks(10, 80)
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=128, window=10,
                     min_count=0, sg=1, workers=8, iter=1)
    model.wv.save_word2vec_format(output_filename)

    return output_filename


def generate_walklets_embeddings(input_file, k):
    """
    Generates Walklets embeddings

    Author: Alexander Ulyanov

    :param input_file: the graph edge list file
    :param k: the scale of the network
    :return: the name of the file where embeddings are saved
    """

    # Read the graph
    G = read_graph(input_file, delimiter=r"\s+")

    # Prepare the output file
    output_folder = "output/embeddings/"
    output_filename = input_file.split(
        "/")[-1].replace(".edgelist", ".walklets")
    output_file = output_folder + output_filename

    # Create and configure Walklets instance
    walklets = Walklets(G, output_file, max_level=k, dimensions=128)

    # Perform random walks with pre-defined skip factor
    walks = walklets.create_walks()

    # Generate embeddings
    embeddings = walklets.create_embeddings(walks)

    # Save embeddings
    walklets.save_model(embeddings)

    # Return the filename where the embeddings are saved
    return output_file


def generate_struc2vec_embeddings(input_file):
    """
    Generates struc2vec embeddings

    Author: Firas Debbichi and Nahla Ben Mosbah
    """
    G = package_1.exec_struc2vec(input_file)
    output = package_1.learn_embeddings(input_file)
    return output


def generate_harp_embeddings(input_file, input_format='edgelist', model='node2vec',
                             number_walks=40, walk_length=10, representation_size=128, window_size=10):
    """
    Generates HARP embeddings

    Author: Oumarou Oumarou
    """

    # Prepare output file
    output_file = 'output/embeddings/' + \
        input_file.split("/")[-1].replace(".edgelist", ".harp")

    # Process args
    if input_format == 'edgelist':
        G = magicgraph.load_edgelist(input_file, undirected=True)
    else:
        raise Exception("Unknown file format: '%s'. Valid formats: 'mat', 'adjlist', and 'edgelist'."
                        % format)
    G = graph_coarsening.DoubleWeightedDiGraph(G)

    if model == 'deepwalk':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G, scale=-1, iter_count=1,
                                                                       num_paths=number_walks, path_length=walk_length,
                                                                       representation_size=representation_size, window_size=window_size,
                                                                       lr_scheme='default', alpha=0.025, min_alpha=0.001, sg=1, hs=1, coarsening_scheme=2, sample=0.1)
    elif model == 'node2vec':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G, scale=-1, iter_count=1,
                                                                       num_paths=number_walks, path_length=walk_length,
                                                                       representation_size=representation_size, window_size=window_size,
                                                                       lr_scheme='default', alpha=0.025, min_alpha=0.001, sg=1, hs=0, coarsening_scheme=2, sample=0.1)
    elif model == 'line':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G, scale=1, iter_count=50,
                                                                       representation_size=64, window_size=1,
                                                                       lr_scheme='default', alpha=0.025, min_alpha=0.001, sg=1, hs=0, sample=0.001)
    savetxt(output_file, embeddings, delimiter=',')
    return output_file


def generate_embeddings(input_file):
    print "-------------------------------------------------"
    print "Generating embeddings for", input_file

    # node2vec
    print "\tgenerating node2vec .. ",
    node2vec_output = generate_node2vec_embeddings(input_file, 1, 1)
    print "Done.\tSaved in: " + str(node2vec_output)

    # Walklets
    print "\tgenerating Walklets .. ",
    walklets_output = generate_walklets_embeddings(input_file, 5)
    print "Done.\tSaved in: " + str(walklets_output)

    # stuc2vec
    print "\tgenerating struc2vec .. ",
    struc2vec_output = generate_struc2vec_embeddings(input_file)
    print "Done.\tSaved in: " + str(struc2vec_output)

    # HARP
    print "\tgenerating HARP .. ",
    harp_output = generate_harp_embeddings(input_file, input_format='edgelist')
    print "Done.\tSaved in: " + str(harp_output)

    print "Completed."


def evaluation():
    datasets = ['cora', 'citeseer']

    print "-------------------------------------------------"
    print "Evaluating embeddings for:", "node2vec"
    ne = Node2VecEval(datasets)
    ne.node2vec_link_prediction()
    ne.node2vec_node_classification()
    ne.node2vec_visualization()

    print "-------------------------------------------------"
    print "Evaluating embeddings for:", "struc2vec"
    ne = Struc2VecEval(datasets)
    ne.struc2vec_link_prediction()
    ne.struc2vec_node_classification()
    ne.struc2vec_visualization()

    print "-------------------------------------------------"
    print "Evaluating embeddings for:", "Walklets"
    we = WalkletsEvaluations(datasets)
    we.walklets_eval_link_prediction()
    we.walklets_eval_node_classification()
    we.walklets_eval_visualization()

    print "-------------------------------------------------"
    print "Evaluating embeddings for:", "HARP"
    hp = Harp_eval(datasets)
    hp.Harp_eval_link_prediction()
    hp.Harp_eval_node_classification()
    hp.Harp_eval_visualization()


def main():
    generate_embeddings("input/cora.edgelist")
    generate_embeddings("input/citeseer.edgelist")
    evaluation()


if __name__ == "__main__":
    main()
