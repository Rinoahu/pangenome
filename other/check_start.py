#!usr/bin/env python
import sys
import numpy as np
import sys

sys.path.append('./pangenome/')

qry = sys.argv[1]

try:
	mode = sys.argv[2]
except:
	mode = 'dbg'

clf = np.load(qry, mmap_mode='r')
if mode == 'dbg':
	x = clf['parameters']
	print('parameters', x)
	print('hash table', x[0]*(8+2+1.)/2**30, 'GB')
	print('elem', x[2]*(8+2+1.)/2**30, 'GB')
	print('load', x[2]*1./x[0])
	print('read', x[-1]/2**30., 'GB')

else:
	from kmer_numba import load_on_disk, dbg2rdbg
	print('prepare loading')
	offset, kmer_dict = load_on_disk(qry, jit=False)
	print('finish loading', kmer_dict.size)
	raise SystemExit()
	rdbg_dict = dbg2rdbg(kmer_dict)
	print('kmer size', rdbg_dict.size, len(rdbg_dict.counts))


	print('keys', clf.keys())
	x = clf['parameters']
	print('parameters', x)
	print('hash table', len(clf['values']))



