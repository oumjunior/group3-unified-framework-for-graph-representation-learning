from embedding_evaluation import *


class Struc2VecEval:

    def __init__(self, data_set_names):
        self.names = data_set_names
        self.edge_paths = dict()
        self.label_paths = dict()
        self.embeddings = dict()
        for x in self.names:
            path = 'output/embeddings/' + x + '.struct2vec'
            self.embeddings[x] = read_embeddings(path)
            self.edge_paths[x] = './input/%s.edgelist' % (x)
            self.label_paths[x] = './input/%s.label' % (x)

    def struc2vec_link_prediction(self):
        for i in self.embeddings.keys():
            link_prediction(
                method="struc2vec", edge_file=self.edge_paths[i], embeddings=self.embeddings[i], name=i, size=128)

    def struc2vec_node_classification(self):
        for i in self.embeddings.keys():
            node_classification(
                method="struc2vec", embeddings=self.embeddings[i], label_path=self.label_paths[i], name=i, size=128)

    def struc2vec_visualization(self):
        for i in self.embeddings.keys():
            plot_embeddings(
                method="struc2vec", embeddings=self.embeddings[i], label_file=self.label_paths[i], name=i)
