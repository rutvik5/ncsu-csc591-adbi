import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def articulations(g, usegraphframe=False):
	# Get the starting count of connected components
	initial_count = g.connectedComponents().select('component').distinct().count()

	# Default version sparkifies the connected components process
	# and serializes node iteration.
	if usegraphframe:
		# Get vertex list for serial iteration
		vertex_list = g.vertices.map(lambda vertex: vertex.id).collect()

		# for vertex in g.vertices.collect():
		# 	vertex_list.append(vertex['id'])

		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
		# the output
		output=[]
		for vertex in vertex_list:
			edges = g.edges.filter("src!='" + vertex + "'").filter("dst!='" + vertex + "'")
			vertices = g.vertices.filter("id!='" + vertex + "'")
			new_gf = GraphFrame(vertices, edges)
			count = new_gf.connectedComponents().select('component').distinct().count()
			if count>initial_count:
				output.append((vertex,1))
			else:
				output.append((vertex,0))

		return sqlContext.createDataFrame(sc.parallelize(output),['id','articulation'])
	# Non-default version sparkifies node iteration and uses networkx
	# for connected components count.
	else:
		vertices =  g.vertices.map(lambda vertex: vertex.id).collect()
		edges = g.edges.map(lambda edge: (edge.src, edge.dst)).collect()
    	G = nx.Graph()
		G.add_nodes_from(vertices)
		G.add_edges_from(edges)
		res=[]
		for vertex in vertices:
			g_temp = G.copy()
			g_temp.remove_node(vertex)
			count = nx.number_connected_components(g_temp)
			if count>initial_count:
				res.append((vertex,1))
			else:
				res.append((vertex,0))
		return sqlContext.createDataFrame(sc.parallelize(res),['id','articulation'])

filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
print("---------------------------")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
