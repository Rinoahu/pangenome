#!usr/bin/env python
import sys
sys.path.append('../')
from kmer_numba import *
from numba import prange
import multiprocessing as mp
import sys



@nb.jit(parallel=True)
def test(hts, n, a, p=8):
    #a = np.random.randint(0, 2**63-1, n)
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
vtype=nb.int8
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


ncpu = mp.cpu_count()
cap = int((N // ncpu) * (1/.75) + 1)
p = ncpu

hts = List()
dct = clf(capacity=cap, ksize=1, ktype=ktype, vsize=1, vtype=vtype)
hts.append(dct)
for i in range(p-1):
    dct = clf(capacity=cap, ksize=1, ktype=ktype, vsize=1, vtype=vtype)
    hts.append(dct)


print('CPU # is', ncpu)

a = np.random.randint(0, 2**31-1, n, dtype='int32')
out = test(hts, N, a, ncpu)

