#!usr/bin/env python
import sys
sys.path.append('../')
from kmer_numba import *
from numba import prange
import multiprocessing as mp
import sys

ncpu = mp.cpu_count()


@nb.jit(parallel=True)
def test(hts, n, p=ncpu):
    a = np.random.randint(0, 2**63-1, n)
    step = int(n // p) + 1
    flag = 0
    for i in prange(p):
        ht = hts[i]
        k, v = np.arange(1), np.arange(1)
        st = i * step
        ed = st + step
        for j in a[st: ed]:
            k[0] = j
            ht.push(k, v)

        print(i, ht.size)
        flag+= ht.size

    print('total N', flag)
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
clf = nb_jitclass(spec)(oakht)


try:
    N = int(eval(sys.argv[1]))
except:
    N = 100

cap = int((N // ncpu) * 1.6)

p = ncpu
hts = List()
dct = clf(capacity=cap, ksize=1, ktype=ktype, vsize=1, vtype=vtype)
hts.append(dct)
for i in range(p-1):
    dct = clf(capacity=cap, ksize=1, ktype=ktype, vsize=1, vtype=vtype)
    hts.append(dct)




print('CPU # is', ncpu)
out = test(hts, N, ncpu)

