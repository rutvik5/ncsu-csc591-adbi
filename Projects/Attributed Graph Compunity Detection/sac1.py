#Unity ID- rkolhe
#Student ID- 200258232

import sys
from igraph import *
from scipy import spatial
import pandas as pd
import numpy as np

def main(alpha):
	
	#read edgelist file and store in a list of tuples
	edges = [] #[(src1, dest1), (src2, dest2), ....]
	with open('data/fb_caltech_small_edgelist.txt') as f:
		lines = f.readlines()
		for line in lines:
			temp = line.strip()
			src = int(temp.split(' ')[0])
			dest = int(temp.split(' ')[1])
			edges.append((src, dest))
			
	#read attributes of each vertex

	attributes = pd.read_csv('data/fb_caltech_small_attrlist.csv')
	
	#create a graph object and add edges and vertices
	g = Graph()
	g.add_vertices(len(attributes))
	g.add_edges(edges)

	#store number of edges and vertices
	total_vertices = g.vcount()
	total_edges = g.ecount()

	#add weight to every edge
	g.es['weight'] = 1.0

	global similarity
	global similarity2
	#create a similarity matrix to store the values
	similarity = np.zeros((total_vertices, total_vertices)) # [[sim00], [sim01], ....[sim0V]
															#  [sim10], [sim12], ....[sim1V]]
	for val in attributes.keys():
		g.vs[val] = attributes[val]

	for i in range(total_vertices):
		for j in range(total_vertices):
			similarity[i][j] = simA(g, i ,j)

	similarity2 = np.array(similarity)
	
	communities = SAC1_p1(alpha, g, g.vcount())

	id=0
	seq_mapp = {}
	tempc = []
	for i in communities:
		if i not in seq_mapp:
			tempc.append(id)
			seq_mapp[i] = id
			id+= 1
			
		else:
			tempc.append(seq_mapp[i])

	communities = tempc

	clusters = list(Clustering(communities))
	sim = 0.0
	QAttr=0
	for cluster in clusters:
		temp = 0.0

		for vertex_1 in cluster:
			for vertex_2 in communities:
				if (vertex_1 != vertex_2):
					temp += similarity[vertex_1][vertex_2]
		temp /= len(cluster)
		sim += temp
	qAttr =  sim/(len(set(communities)))

	composite_mod_1 = g.modularity(communities, weights='weight') + qAttr

	#start phase 2
	SAC1_p2(communities, g)

	#phase1- iter2
	communities2 = SAC1_p1(alpha, g, g.vcount())
	
	id=0
	seq_mapp = {}
	tempc = []
	for i in communities2:
		if i not in seq_mapp:
			tempc.append(id)
			seq_mapp[i] = id
			id+= 1
			
		else:
			tempc.append(seq_mapp[i])

	new_communities2 = tempc

	clusters2 = list(Clustering(new_communities2))
	
	clusters = list(Clustering(communities))
	sim = 0.0
	QAttr=0
	for cluster in clusters:
		temp = 0.0

		for vertex_1 in cluster:
			for vertex_2 in communities:
				if (vertex_1 != vertex_2):
					temp += similarity[vertex_1][vertex_2]
		temp /= len(cluster)
		sim += temp
	qAttr2 = sim/(len(set(communities)))

	composite_mod_2 = g.modularity(communities, weights='weight') + qAttr2

	id=0
	seq_mapp = {}
	tempc = []
	for i in communities:
		if i not in seq_mapp:
			tempc.append(id)
			seq_mapp[i] = id
			id+= 1
			
		else:
			tempc.append(seq_mapp[i])

	communities1 = tempc
	clusters_p1 = list(Clustering(communities1))
	communitiesf = list()

	for cluster in clusters2:
		temp = list()
		for vertex in cluster:
			temp.extend(clusters_p1[vertex])
		communitiesf.append(temp)

	if (composite_mod_1 < composite_mod_2):
		write_file(clusters2, alpha)
		
	else:
		write_file(clusters_p1, alpha)

def write_file(clusters, alpha):
	if alpha == 0.5:
		alpha = 5
	fileName = "communities_" + str(int(alpha)) + ".txt"

	with open(fileName, 'w') as f:
	    for cluster in clusters:
	    	for i in range(len(cluster)-1):
	    		f.write("%s," % cluster[i])
	    	f.write(str(cluster[-1]))
	    	f.write('\n')

def SAC1_p2(communities, g):
	id=0
	seq_mapp = {}
	tempc = []
	for i in communities:
		if i not in seq_mapp:
			tempc.append(id)
			seq_mapp[i] = id
			id+= 1
			
		else:
			tempc.append(seq_mapp[i])

	new_communities = tempc

	temp = list(Clustering(new_communities))
	size_new_comms = len(set(new_communities))
	similarity = np.zeros((size_new_comms,size_new_comms))
	
	
	for i in range(size_new_comms):
		for j in range(size_new_comms):
			sim = 0.0
			for k in temp[i]:
				for l in temp[j]:
					sim += similarity2[k][l]
			similarity[i][j] = sim
	
	g.contract_vertices(new_communities)
	g.simplify(combine_edges=sum)
	return	            

def SAC1_p1(alpha, g, total_vertices):
	comms = list(range(total_vertices))
	iterations = 0
	flag = False
	while iterations<15 and flag== False:
			flag = True
			for j in range(len(g.vs)):
				max_vertex = -1
				max_dQ = 0.0		
				clusters = list(set(comms))
				for k in clusters:
					if comms[j] != k:
						Q1 = g.modularity(comms, weights='weight')
						temp = comms[j]
						comms[j] = k
						Q2 = g.modularity(comms, weights='weight')
						comms[j] = temp
						d1 = Q2-Q1

						sims = 0.0;
						indices = [i for i, x in enumerate(comms) if x == k]
						for vertex in indices:
							sims += similarity[j][vertex]
						d2 = sims/(len(indices)*len(set(comms)))

						dQ = (alpha*d1) + ((1-alpha)*d2)

						if dQ > max_dQ:
							max_dQ = dQ
							max_vertex = k
				if max_dQ > 0.0 and max_vertex != 0:
					flag = False
					comms[j] = max_vertex
			iterations +=1
	return comms


def simA(g, i, j):
	vertex1 = list(g.vs[i].attributes().values())
	vertex2 = list(g.vs[j].attributes().values())

	return 1-spatial.distance.cosine(vertex1, vertex2)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print ('Invalid Input')
		sys.exit(0)
	else:
		alpha = float(sys.argv[1])
		main(alpha)