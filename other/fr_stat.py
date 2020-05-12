#!usr/bin/env python
import sys
import numpy as np

qry = sys.argv[1]

frs = []
f = open(qry, 'r')
for i in f:
	if '+' not in i:
		continue
	j = i[:-1].split('\t')
	st, ed = map(int, j[1:3])
	frs.append(abs(ed-st))

f.close()

#print(np.mean(frs))
print('num', len(frs))
print('max', max(frs))
print('avg', sum(frs)*1./len(frs))
