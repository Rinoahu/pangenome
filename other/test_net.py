import networkx as nx

g = nx.Graph()

f = open('Cs_genome_v2.fa_rdbg_weight.xyz', 'r')
for i in f:
	j, k = i.split('\t')[:2]
	g.add_edge(j, k)


for i in nx.connected_components(g):
	print('size\t%d'%len(i))
