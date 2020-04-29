from embedding_evaluation import *


def read_embeddings(file_name):

    # read all the lines
    f = open(file_name, "r")
    lines = f.readlines()
    f.close()

    # detemine how many representations of the graph are present
    num_cols = len(lines[0].split(","))
    num_dimensions = 128
    max_k = num_cols / num_dimensions

    # initialize the destination dictionary
    embeddings = {}
    for i in range(max_k):
        embeddings[i + 1] = {}
        for n in range(len(lines) - 1):
            embeddings[i + 1][str(n)] = []

    # parse and add embeddings
    for i in range(1, len(lines)):
        line = lines[i]
        parts = line.split(",")

        for k in range(max_k):
            emb = []
            for d in range(num_dimensions):
                emb.append(float(parts[128 * k + d]))
            embeddings[k + 1][str(i - 1)] = emb

    return embeddings


class WalkletsEvaluations:
    def __init__(self, data_set_names):
        self.names = data_set_names
        self.edge_paths = dict()
        self.label_paths = dict()
        self.embeddings = dict()

        for x in self.names:
            path = 'output/embeddings/' + x + '.walklets'
            self.embeddings[x] = read_embeddings(path)

            self.edge_paths[x] = './input/%s.edgelist' % (x)
            self.label_paths[x] = './input/%s.label' % (x)

    def walklets_eval_link_prediction(self):
        for i in self.embeddings.keys():
            for k in self.embeddings[i].keys():
                print '\tK =', k
                link_prediction(
                    method="walklets_k" + str(k),
                    edge_file=self.edge_paths[i],
                    embeddings=self.embeddings[i][k],
                    name=i, size=128)

    def walklets_eval_node_classification(self):
        for i in self.embeddings.keys():
            for k in self.embeddings[i].keys():
                print '\tK =', k
                node_classification(
                    method="walklets_k" + str(k),
                    embeddings=self.embeddings[i][k],
                    label_path=self.label_paths[i],
                    name=i, size=128)

    def walklets_eval_visualization(self):
        for i in self.embeddings.keys():
            for k in self.embeddings[i].keys():
                print '\tK =', k
                plot_embeddings(
                    method="walklets_k" + str(k),
                    embeddings=self.embeddings[i][k],
                    label_file=self.label_paths[i],
                    name=i)
