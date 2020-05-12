#!usr/bin/env python
import networkx as nx
import sys
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

G = nx.DiGraph()
qry = sys.argv[1]

f = open(qry, 'r')
for i in f:
	j = i[:-1].split('\t')
	x, y = j[:2]
	G.add_edge(x, y)

nx.draw(G)  # networkx draw()
#plt.draw()  # pyplot draw()
plt.savefig('net.png', format='png')

