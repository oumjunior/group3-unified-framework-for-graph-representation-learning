# -*- coding: utf-8 -*-
import logging
import numpy as np
import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
import graph

logging.basicConfig(filename='struc2vec.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

def read_graph(input_file):
	'''
	Reads the input network.
	'''
	logging.info(" - Loading graph...")
	G = graph.load_edgelist(input_file,undirected=True)
	logging.info(" - Graph loaded.")
	return G

def learn_embeddings(input_file):

        input_file = input_file.replace(input_file[:6], '')
        output = 'output/embeddings/' + input_file.replace('.edgelist','.struct2vec')
        


	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	logging.info("Initializing creation of the representations...")
	walks = LineSentence('random_walks.txt')
	model = Word2Vec(walks, size=128, window=10, min_count=0, hs=1, sg=1, workers=4, iter=5)
	model.wv.save_word2vec_format(output)
	logging.info("Representations created.")

	return output

def exec_struc2vec(input_file):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''

	until_layer = 6


	G = read_graph(input_file)
	G = struc2vec.Graph(G, 'undirected', workers = 4 , untilLayer = until_layer)


	G.preprocess_neighbors_with_bfs_compact()



	G.create_vectors()
	G.calc_distances(compactDegree = True)



	G.create_distances_network()
	G.preprocess_parameters_random_walk()

	G.simulate_walks(10, 80)


	return G
