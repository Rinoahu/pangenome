#!usr/bin/env python
import sys
sys.path.append('../')
from kmer_numba import *
from numba import prange

@nb.jit(parallel=True)
def test(hts, n, p=8):
    a = np.random.randint(0, 2**63-1, n)
    step = int(n // p) + 1
    for i in prange(p):
        ht = hts[i]
        k, v = np.arange(1), np.arange(1)
        st = i * step
        ed = st + step
        for j in a[st: ed]:
            k[0] = j
            ht.push(k, v)

        print(i, ht.size)
    return a

ktype=nb.int32
vtype=nb.int32
spec = {}
spec['capacity'] = nb.int64
spec['load'] = nb.float32
spec['size'] = nb.int64
spec['ksize'] = nb.int64
spec['vsize'] = nb.int64
spec['keys'] = ktype[:]
spec['values'] = vtype[:]
spec['counts'] = nb.uint8[:]
clf = nb.jitclass(spec)(oakht)


p = 8
hts = List()
dct = clf(capacity=2**20, ksize=1, ktype=ktype, vsize=1, vtype=vtype)
hts.append(dct)
for i in range(p-1):
    dct = clf(capacity=2**20, ksize=1, ktype=ktype, vsize=1, vtype=vtype)
    hts.append(dct)


#print(clf)

import sys
try:
    N = int(eval(sys.argv[1]))
except:
    N = 100
out = test(hts, N)

