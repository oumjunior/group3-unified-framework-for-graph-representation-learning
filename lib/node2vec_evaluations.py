from embedding_evaluation import *


class Node2VecEval:

    def __init__(self, data_set_names):
        self.names = data_set_names
        self.edge_paths = dict()
        self.label_paths = dict()
        self.embeddings = dict()
        for x in self.names:
            path = 'output/embeddings/' + x + '.node2vec'
            self.embeddings[x] = read_embeddings(path)
            self.edge_paths[x] = './input/%s.edgelist' % (x)
            self.label_paths[x] = './input/%s.label' % (x)

    def node2vec_link_prediction(self):
        for i in self.embeddings.keys():
            link_prediction(
                method="node2vec", edge_file=self.edge_paths[i], embeddings=self.embeddings[i], name=i, size=128)

    def node2vec_node_classification(self):
        for i in self.embeddings.keys():
            node_classification(
                method="node2vec", embeddings=self.embeddings[i], label_path=self.label_paths[i], name=i, size=128)

    def node2vec_visualization(self):
        for i in self.embeddings.keys():
            plot_embeddings(
                method="node2vec", embeddings=self.embeddings[i], label_file=self.label_paths[i], name=i)
