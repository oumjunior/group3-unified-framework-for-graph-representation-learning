#########################################################################################
from collections import OrderedDict
import sys

import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

VERBOSE = False


class TopKRanker(OneVsRestClassifier):

    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf, name):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.name = name

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)

        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)

        results['acc'] = accuracy_score(Y, Y_)

        # print('-------------------')
        # print(self.name, 'node calssification: ',  results)
        if VERBOSE:
            print "\t\t`{}` node classification results:".format(self.name)
            for metric, result in results.items():
                print "\t\t\t{:10}{}".format(metric, result)

        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def read_node_label(embeddings=None, label_file=None, skip_head=False):

    fin = open(label_file, 'r')
    X = []
    Y = []
    label = {}

    for line in fin:
        a = line.strip('\n').split(' ')
        label[a[0]] = a[1]

    fin.close()
    for i in embeddings:
        X.append(i)
        Y.append(label[str(i)])

    return X, Y


def print_results(dataset_name, stats):
    print "\tNode classification results for `{}`".format(dataset_name)

    if not stats:
        "No results!"
        return

    pattern = "\t{:10}" + "|" + ("{:>4}|" * len(stats))

    # we assume that all the results have the same metrics
    metrics = stats.values()[0].keys()

    # print header
    cols = ["{:2.0f}%".format(x * 100) for x in stats.keys()]
    header = pattern.format("", *cols)
    line = "\t" + ("-" * len(header))
    print header
    print line

    for metric in metrics:
        values = []
        for _, v in stats.items():
            value = v.get(metric)
            values.append("{:.2f}".format(value))
        print pattern.format(metric, *values)

    print line


def node_classification(method, embeddings, label_path, name, size):

    X, Y = read_node_label(embeddings, label_path,)

    f_c = open('./results/%s_%s_classification_%d.txt' %
               (method, name, size), 'w')

    stats = OrderedDict()

    all_ratio = []

    for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        if VERBOSE:
            print("\tTraining classifier using {:.2f}% nodes...".format(
                tr_frac * 100))
        clf = Classifier(embeddings=embeddings,
                         clf=LogisticRegression(solver='liblinear'), name=name)
        results = clf.split_train_evaluate(X, Y, tr_frac)

        stats[tr_frac] = results

        avg = 'macro'
        f_c.write(name + ' train percentage: ' + str(tr_frac) +
                  ' F1-' + avg + ' ' + str('%0.5f' % results[avg]))
        all_ratio.append(results[avg])
        f_c.write('\n')

    print_results(name, stats)


####################################################################################

import networkx as nx
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class Graph():

    def __init__(self,
                 nx_G=None, is_directed=False,
                 prop_pos=0.5, prop_neg=0.5,
                 workers=1,
                 random_seed=None):
        self.G = nx_G
        self.is_directed = is_directed
        self.prop_pos = prop_pos
        self.prop_neg = prop_neg
        self.wvecs = None
        self.workers = workers
        self._rnd = np.random.RandomState(seed=random_seed)

    def read_graph(self, input):

        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G.adj[edge[0]][edge[1]]['weight'] = 1

        print "\tReading graph `{}` (nodes: {}, edges: {})".format(
            input.split("/")[-1],
            G.number_of_nodes(),
            G.number_of_edges()
        )

        G1 = G.to_undirected()
        self.G = G1

    def generate_pos_neg_links(self):

        # Select n edges at random (positive samples)
        n_edges = self.G.number_of_edges()
        n_nodes = self.G.number_of_nodes()
        npos = int(self.prop_pos * n_edges)
        nneg = int(self.prop_neg * n_edges)

        n_neighbors = [len(list(self.G.neighbors(v))) for v in self.G.nodes()]
        n_non_edges = n_nodes - 1 - np.array(n_neighbors)

        non_edges = [e for e in nx.non_edges(self.G)]
        if VERBOSE:
            print("\tFinding %d of %d non-edges" % (nneg, len(non_edges)))

        # Select m pairs of non-edges (negative samples)
        rnd_inx = self._rnd.choice(len(non_edges), nneg, replace=False)
        neg_edge_list = [non_edges[ii] for ii in rnd_inx]

        if len(neg_edge_list) < nneg:
            raise RuntimeWarning(
                "\tOnly %d negative edges found" % (len(neg_edge_list))
            )

        if VERBOSE:
            print("\tFinding %d positive edges of %d total edges" %
                  (npos, n_edges))

        # Find positive edges, and remove them.
        edges = self.G.edges()

        edges = list(edges)

        pos_edge_list = []
        n_count = 0
        n_ignored_count = 0

        rnd_inx = self._rnd.permutation(n_edges)

        for eii in rnd_inx.tolist():
            edge = edges[eii]

            # Remove edge from graph
            data = self.G[edge[0]][edge[1]]
            self.G.remove_edge(*edge)

            reachable_from_v1 = nx.connected._plain_bfs(self.G, edge[0])
            if edge[1] not in reachable_from_v1:
                self.G.add_edge(*edge, **data)
                n_ignored_count += 1
            else:
                pos_edge_list.append(edge)
                if VERBOSE:
                    sys.stdout.write(
                        "\r" + "\tFound: {} edges".format(n_count + 1))
                n_count += 1

            if n_count >= npos:
                break

        if VERBOSE:
            sys.stdout.write("\n")

        edges_num = len(pos_edge_list)
        self._pos_edge_list = pos_edge_list
        self._neg_edge_list = neg_edge_list

        # print('pos_edge_list', len(self._pos_edge_list))
        # print('neg_edge_list', len(self._neg_edge_list))
        if VERBOSE:
            print("\tEdge list lengths: Pos: {} Neg: {}".format(
                len(self._pos_edge_list), len(self._neg_edge_list)))

    def get_selected_edges(self):
        edges = self._pos_edge_list + self._neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(self._pos_edge_list)] = 1
        return edges, labels

    def edges_to_features(self, edge_list, edge_function, emb_size, model):

        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, emb_size), dtype='f')

        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = np.asarray(model[str(v1)])
            emb2 = np.asarray(model[str(v2)])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec


def create_train_test_graphs(input='Facebook.edges', workers=8):

    default_params = {
        'edge_function': "hadamard",
        # Proportion of edges to remove nad use as positive samples
        "prop_pos": 0.5,
        "prop_neg": 0.5,                # Number of non-edges to use as negative samples

    }

    # Remove half the edges, and the same number of "negative" edges
    prop_pos = default_params['prop_pos']
    prop_neg = default_params['prop_neg']

    # print("Regenerating link prediction graphs")
    # Train graph embeddings on graph with random links
    Gtrain = Graph(is_directed=False,
                   prop_pos=prop_pos,
                   prop_neg=prop_neg,
                   workers=workers)
    Gtrain.read_graph(input)
    Gtrain.generate_pos_neg_links()

    # Generate a different random graph for testing
    Gtest = Graph(is_directed=False,
                  prop_pos=prop_pos,
                  prop_neg=prop_neg,
                  workers=workers)
    Gtest.read_graph(input)
    Gtest.generate_pos_neg_links()

    return Gtrain, Gtest


def test_edge_functions(num_experiments=2, emb_size=128, model=None, edges_train=None, edges_test=None, Gtrain=None, Gtest=None, labels_train=None, labels_test=None):

    edge_functions = {
        "hadamard": lambda a, b: a * b,
        "average": lambda a, b: 0.5 * (a + b),
        "l1": lambda a, b: np.abs(a - b),
        "l2": lambda a, b: np.abs(a - b) ** 2,
    }

    aucs = {func: [] for func in edge_functions}

    print "\tRunning experiments"

    for iter in range(num_experiments):

        if VERBOSE:
            print("\tIteration %d of %d" % (iter + 1, num_experiments))

        for edge_fn_name, edge_fn in edge_functions.items():

            # print(edge_fn_name, edge_fn)
            # Calculate edge embeddings using binary function
            edge_features_train = Gtrain.edges_to_features(
                edges_train, edge_fn, emb_size, model)
            edge_features_test = Gtest.edges_to_features(
                edges_test, edge_fn, emb_size, model)

            # Linear classifier
            scaler = StandardScaler()
            lin_clf = LogisticRegression(C=1, solver='liblinear')

            clf = pipeline.make_pipeline(scaler, lin_clf)

            # Train classifier
            clf.fit(edge_features_train, labels_train)
            AUC = metrics.scorer.roc_auc_scorer(
                clf, edge_features_test, labels_test)
            aucs[edge_fn_name].append(AUC)

    return aucs


def prediction(edge_file, embeddings, size):

    Gtrain, Gtest = create_train_test_graphs(edge_file, workers=2)

    # Train and test graphs, with different edges
    edges_train, labels_train = Gtrain.get_selected_edges()
    edges_test, labels_test = Gtest.get_selected_edges()

    auc = test_edge_functions(2, size, embeddings, edges_train,
                              edges_test, Gtrain,  Gtest, labels_train, labels_test)

    return auc


def link_prediction(method, edge_file, embeddings, name, size):

    auc = prediction(edge_file, embeddings, size)

    functions = ["hadamard", "average", "l1", "l2"]

    f_l = open('./results/%s_%s_linkpred_%d.txt' % (method, name, size), 'w')

    print "\tLink prediction results for `{}`:".format(name)
    for i in functions:
        print "\t\taggr: {:10}score: {:.3f}".format(i, np.mean(auc[i]))

        f_l.write(name + ' ' + str(size) + ' ' + str(i) +
                  ' link-pred AUC: ' + str('%.3f' % np.mean(auc[i])))
        f_l.write('\n')


##################################################################################################################
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_embeddings(method, embeddings, label_file, name):

    plt.figure()

    print "\tPlotting embeddings for `{}`".format(name)

    X, Y = read_node_label(embeddings, label_file)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1],
                    label=c)  # c=node_colors)
    plt.legend()

    plt.savefig('./results/%s_%s.png' % (method, name))  # or '%s.pdf'%name
    # plt.show()

##########################################################################################################################


def read_embeddings(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    embeddings = {}
    for i in range(1, len(lines)):
        parts = lines[i].split(" ")
        node_num = parts[0]
        values = [float(x) for x in parts[1:]]
        embeddings[node_num] = values

    return embeddings


def to_dict(embeddings):
    if type(embeddings) == dict:
        return embeddings
    dic_embeddings = dict()
    for i in range(len(embeddings)):
        dic_embeddings[str(i)] = embeddings[i]
    return dic_embeddings
