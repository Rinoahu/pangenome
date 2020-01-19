# pangenome
pangenome methods for large-scale sequencing data. Of course, it is proof-of-concept

# kmer_pypy.py
1. Build a dBG
2. Build reduced dBG by removing nodes with <= 1 indegree and outdegree in dBG
3. Convert sequence to compressed path according to 2
4. Remove weak edges in dBG and label each connect component of the rdBG
5. label each sequence again
