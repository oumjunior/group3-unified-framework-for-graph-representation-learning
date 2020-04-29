import random

from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd

class Walklets:
	"""
	Simplified and pre-configures adaptation of the code from:

	Don't Walk, Skip! Online Learning of Multi-scale Network Embeddings.
	Bryan Perozzi, Vivek Kulkarni, Haochen Chen, Steven Skiena. ASONAM, 2017.
	https://arxiv.org/abs/1605.02115

	Github link to the original code: https://github.com/benedekrozemberczki/walklets

	Simplified, post-configured and commented by: Alexander Ulyanov
	"""


	def __init__(self, graph, output_file,
		dimensions=128, max_level=5, num_walks_from_source=5, walk_length=80):
		self.graph = graph
		self.output_file = output_file
		self.dimensions = dimensions
		self.max_level = max_level
		self.num_walks_from_source = num_walks_from_source
		self.walk_length = walk_length


	def add_skips_to_walk(self, walk, length):
		walk_with_skips = []

		for step in range(length + 1):
			neighbors = [y for i, y in enumerate(walk[step:])
						 if i % length == 0]

			walk_with_skips.append(neighbors)

		return walk_with_skips


	def create_embeddings(self, walks):
		"""
		Creates embeddings for the desired level of network from complete random walks.
		"""

		embeddings = []

		# Repeat for each power of the adjacency matrix
		for level in range(1, self.max_level+1):

			# From the list of full walks, add skips of a pre-set length
			repr_at_level = self.create_skips_in_walks(walks, level)

			# Feed the random walks with a skip-factor into Word2Vec Skip-gram algorithm
			model = Word2Vec(repr_at_level, size=self.dimensions, window=1, min_count=1, sg=1, workers=4)

			# Retrieve produced embeddings
			new_embedding = self.get_embedding(model)

			# Append to the full list of embeddings of different levels
			embeddings = embeddings + [new_embedding]

		# The embeddings of different levels of the network are separate lists
		# Flattening the list will produce a matrix
		# where for each node, all embeddings will be presented
		# in the ascending order of powers of matrix A
		embeddings = np.concatenate(embeddings, axis=1)

		return embeddings


	def create_skips_in_walks(self, walks, skip_length):
		"""
		Adds skips of the specified length to the walks in the list.
		"""

		walks_with_skips = [self.add_skips_to_walk(walk, skip_length) for walk in walks]
		walks_with_skips = [w for walks in walks_with_skips for w in walks]

		return walks_with_skips


	def create_walks(self):
		"""
		Creates random walks without skipping any nodes.
		"""

		walks = []

		# Loop to create the desired number of random walks from one source
		for walk_num in range(self.num_walks_from_source):

			# Random walk from each node in the graph
			for node in self.graph.nodes():
				walk_from_node = self.walk(node)
				walks.append(walk_from_node)

		return walks


	def get_embedding(self, model):
		"""
		Retrieves the representation of the nodes in a network
		"""
		embedding = []
		for node in range(len(self.graph.nodes())):
			embedding.append(list(model[str(node)]))
		embedding = np.array(embedding)
		return embedding


	def save_model(self, embeddings):
		"""
		Saves the embeddings to the CSV file.
		"""
		column_names = ["x_" + str(x) for x in range(embeddings.shape[1])]
		self.embedding = pd.DataFrame(embeddings, columns=column_names)
		self.embedding.to_csv(self.output_file, index=None)


	def walk(self, node):
		"""
		Performs the pre-defined number of random walks from the source node.
		"""

		# Add the source node to the walk
		walk = [node]

		# Repeat for the desired number of node in a walk
		# Subtract one, because the source node is already in the walk.
		for _ in range(self.walk_length-1):

			# Collect the neighbors
			nebs = [node for node in self.graph.neighbors(walk[-1])]

			# If neighbors exist
			if len(nebs) > 0:

				# Randomly select 1 neighbor and append it to the walk
				walk = walk + random.sample(nebs, 1)

		walk = [str(w) for w in walk]

		return walk
