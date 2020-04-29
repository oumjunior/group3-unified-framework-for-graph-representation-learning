from embedding_evaluation import *


class Harp_eval:
    def __init__(self, data_set_names):
        self.names = data_set_names
        self.edge_paths = dict()
        self.label_paths = dict()
        self.embeddings = dict()
        for x in self.names:
            path = 'output/embeddings/' + x + '.harp'
            self.embeddings[x] = np.loadtxt(path, delimiter=",")
            self.embeddings[x] = to_dict(self.embeddings[x])
            self.edge_paths[x] = './input/%s.edgelist' % (x)
            self.label_paths[x] = './input/%s.label' % (x)

    def Harp_eval_link_prediction(self):
        for i in self.embeddings.keys():
            link_prediction(
                method="HARP", edge_file=self.edge_paths[i], embeddings=self.embeddings[i], name=i, size=128)

    def Harp_eval_node_classification(self):
        for i in self.embeddings.keys():
            node_classification(
                method="HARP", embeddings=self.embeddings[i], label_path=self.label_paths[i], name=i, size=128)

    def Harp_eval_visualization(self):
        for i in self.embeddings.keys():
            plot_embeddings(
                method="HARP", embeddings=self.embeddings[i], label_file=self.label_paths[i], name=i)
