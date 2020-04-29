import networkx as nx
import pandas as pd

def read_graph(file_name, delimiter=None):
	"""
	Read the graph from the edgelist file
	"""
	edges = pd.read_csv(file_name, delimiter=delimiter).values.tolist()
	graph = nx.from_edgelist(edges)
	return graph
