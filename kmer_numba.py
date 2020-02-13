#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# proof of concept

import sys
from Bio import SeqIO
from array import array
from bisect import bisect_left
import math
import mmap
import networkx as nx
import gc
import itertools
import os
import platform
from random import randint

try:
    from _numpypy import multiarray as np
except:
    import numpy as np

#try:
#    from numba import njit, jitclass
#except:
#    njit = lambda x: x
#    jitclass = njit

try:
    from numba.typed import Dict, List
except:
    Dict, List = dict, list

try:
    import numba as nb
except:
    class NB:
        def __init__(self):
            self.uint64 = 'uint64'
            self.uint32 = 'uint32'
            self.uint16 = 'uint16'
            self.uint8 = 'uint8'

            self.int64 = 'int64'
            self.int32 = 'int32'
            self.int16 = 'int16'
            self.int8 = 'int8'

            self.float64 = 'float64'
            self.float32 = 'int32'
            self.float16 = 'float16'
            self.float = 'float32'

            self.longlong = int
            self.ulonglong = int

            self.njit = lambda x: x
            self.jitclass = lambda x: x

    nb = NB()

try:
    xrange = xrange
except:
    xrange = range

# memmap function for pypy
def memmap(fn, mode='w+', shape=None, dtype='int8'):
    if dtype == 'int8' or dtype == 'uint8':
        stride = 1
    elif dtype == 'float16' or dtype == 'int16' or dtype == 'uint16':
        stride = 2
    elif dtype == 'float32' or dtype == 'int32' or dtype == 'uint32':
        stride = 4
    else:
        stride = 8

    if isinstance(shape, int):
        L = shape
    elif isinstance(shape, tuple): 
        L = 1
        for i in shape:
            L *= i
    else:
        L = 0

    if 'w' in mode and L > 0:
        f = open(fn, mode)
        f.seek(L*stride-1)
        f.write('\x00')
        f.seek(0)
    else:
        f = open(fn, mode)

    #print 'L', L
    buf = mmap.mmap(f.fileno(), L*stride, prot=mmap.ACCESS_WRITE)
    #return np.frombuffer(buf, dtype=dtype).reshape(shape), f
    return np.frombuffer(buf, dtype=dtype), f

# jit version
class oakht:
    def __init__(self, capacity=1024, load_factor = .75, ksize=1, ktype=nb.int64, vtype=nb.int64):

        self.capacity = self.find_prime(capacity)
        self.load = load_factor
        self.size = 0
        self.ksize = ksize
        N = self.capacity

        self.keys = np.empty(N * ksize, dtype=ktype)
        self.values = np.empty(N, dtype=vtype)
        self.counts = np.zeros(N, dtype=nb.uint8)

    # check a num is prime
    def isprime(self, n) : 
        if n <= 1 or n % 2 == 0 or n % 3 == 0: 
            return False
        if n == 2 or n == 3:
            return True
        i = 5
        while(i * i <= n) : 
            if n % i == 0 or n % (i + 2) == 0: 
                return False
            i += 6
  
        return True

    # find the minimum prime that >= n
    def find_prime(self, n):
        for i in range(n, n+7*10**7):
            if self.isprime(i):
                return i

    # clear the dict
    def clear(self):
        self.counts[:] = 0

    # whether key0 == key1
    def eq(self, k0, s0, k1, s1, N):
        for i in xrange(N):
            if k0[s0+i] != k1[s1+i]:
                return False

        return True

    def fnv(self, data, start=0, end=0):
        a, b, c = nb.ulonglong(0xcbf29ce484222325), nb.ulonglong(0x100000001b3), nb.ulonglong(0xffffffffffffffff)
        if end - start == 1:
            #val = nb.ulonglong(data[start])
            val = data[start]
            for i in xrange(4):
                s = nb.ulonglong(val & 0b11111111)
                a ^= s
                a *= b
                a &= c
                #val = nb.ulonglong(val >> 2)
                val = (val >> 8)
        else:
            for i in xrange(start, end):
                s = nb.ulonglong(data[i])
                a ^= s
                a *= b
                a &= c
        return a

    def hash_(self, data, start=0, size=1):
        return self.fnv(data, start, start + size)

    def resize(self):

        # get old arrays
        N = self.capacity
        ks = self.ksize
        keys_old, values_old, counts_old = self.keys, self.values, self.counts

        # get new arrays
        self.capacity = self.find_prime(nb.longlong(N * 1.62))
        M = self.capacity

        keys = np.empty(M * ks, dtype=keys_old.dtype)
        values = np.empty(M, dtype=values_old.dtype)
        counts = np.zeros(M, dtype=counts_old.dtype)

        for i in xrange(N):
            if counts_old[i] > 0:
                value = values_old[i]
                count = counts_old[i]
                ik = i * ks
                # new hash
                j, k = self.hash_(keys_old, ik, ks) % M, 0
                j_init = j
                for k in xrange(N):
                    jk = j * ks

                    if counts[j] == 0 or self.eq(keys, jk, keys_old, ik, ks):
                        break

                    j = (j_init + k * k) % M

                jk = j * ks
                keys[jk: jk+ks] = keys_old[ik: ik+ks]
                values[j] = value
                counts[j] = count

            else:
                continue

        self.keys = keys
        self.values = values
        self.counts = counts

        del keys_old, values_old, counts_old

        #gc.collect()

    def pointer(self, key, start=0):
        ks = self.ksize
        M = self.capacity

        #j, k = hash(key) % M, 0
        j, k = self.hash_(key, start, ks) % M, 0

        k = 0
        j_init = j
        for k in xrange(M):
            jk = j * ks
            if self.eq(self.keys, jk, key, start, ks) or self.counts[j] == 0:
                #print('depth', k)
                break

            j = (j_init + k * k) % M

        return j

    def push(self, key, value, start=0):
        j = self.pointer(key, start)
        ks = self.ksize
        jk = j * ks
        if self.counts[j] == 0:
            self.size += 1
            self.keys[jk: jk+ks] = key[start: start+ks]

        self.values[j] = value
        self.counts[j] = min(self.counts[j] + 1, 255)

        # if too many elements
        if self.size * 1. / self.capacity > self.load:
            a, b =  self.size, self.capacity
            self.resize()
            #print('before resize', a, b, 'after resize', self.size, self.capacity)

    def __setitem__(self, key, value, start=0):
        self.push(key, value, start)

    def get(self, key, start=0):
        j = self.pointer(key, start)
        ks = self.ksize
        jk = j * ks
        if self.eq(self.keys, jk, key, start, ks):
            return self.values[j]
        else:
            raise KeyError

    def __getitem__(self, key, start=0):
        self.get(key, start)

    def get_count(self, key, start=0):
        j = self.pointer(key)
        ks = self.ksize
        jk = j * ks
        if self.eq(self.keys, jk, key, start, ks):
            return self.counts[j]
        else:
            return 0

    def __delitem__(self, key, start=0):
        j = self.pointer(key)
        ks = self.ksize
        jk = j * ks
        if self.eq(self.keys, jk, key, start, ks):
            self.counts[j] = 0
            self.size -= 1
        else:
            raise KeyError

    def has_key(self, key, start=0):
        j = self.pointer(key)
        ks = self.ksize
        jk = j * ks
        return self.eq(self.keys, jk, key, start, ks)
   
    def iterkeys(self):
        ks = self.ksize
        for i in xrange(0, self.counts.shape[0]):
            ik = i * ks
            if self.counts[i] > 0:
                yield self.keys[ik:ik+ks]

    def __iter__(self):
        self.iterkeys()

    def itervalues(self):
        for i in xrange(0, self.counts.shape[0]):
            if self.counts[i] > 0:
                yield self.values[i]


    def iteritems(self):
        ks = self.ksize
        for i in xrange(0, self.counts.shape[0]):
            ik = i * ks
            if self.counts[i] > 0:
                yield self.keys[ik:ik+ks], self.values[i]

    def len(self):
        return self.size

    def __len__(self):
        return self.size

    # save hash table to disk
    def dump(self, fname):
        M = self.keys.shape
        key_type, val_type = self.ktype, self.vtype
        dump_keys, dump_fk = memmap(fname + '_dump_key.npy', shape=M, dtype=key_type)
        dump_keys[:] = self.keys
        dump_fk.close()

        N = self.values.shape
        dump_values, dump_fv = memmap(fname + '_dump_val.npy', shape=N, dtype=val_type)
        dump_values[:] = self.values
        dump_fv.close()

        dump_counts, dump_fc = memmap(fname + '_dump_cnt.npy', shape=N, dtype='uint8')
        dump_counts[:] = self.counts
        dump_fc.close()

    # load hash table from disk
    def loading(self, fname):

        key_type, val_type = self.ktype, self.vtype

        dump_keys, dump_fk = memmap(fname + '_dump_key.npy', 'a+', dtype=key_type)
        self.keys = np.array(dump_keys)
        dump_fk.close()

        dump_values, dump_fv = memmap(fname + '_dump_val.npy', 'a+', dtype=val_type)
        self.values = np.array(dump_values)
        dump_fv.close()

        dump_counts, dump_fc = memmap(fname + '_dump_cnt.npy', 'a+', dtype='uint8')
        self.counts = np.array(dump_counts)
        dump_fc.close()

        self.ksize = len(self.keys) // len(self.values)

        capacity  = len(self.counts)
        self.primes = [elem for elem in primes if elem >= capacity]
        self.capacity = self.primes.pop()

        self.size = (self.counts > 0).sum()


# combine several dict
class mdict:
    def __init__(self, capacities=1024, load_factor=.75, key_type='uint64', val_type='uint16', fuc=oakht, bucket=32):
        self.blk = bucket
        self.ktype = key_type
        self.vtype = val_type
        self.load = load_factor
        self.capacity = capacities
        N = self.capacity // self.blk + 1
        self.dicts = [fuc(capacity=N, load_factor=self.load, key_type=self.key_type, val_type=self.val_type, disk=False) for elem in xrange(self.blk)]

    def __setitem__(self, key, value):
        i = hash(key) % self.blk
        self.dicts[i][key] = value

    def __getitem__(self, key):
        i = hash(key) % self.blk
        return self.dicts[i][key]

    def __delitem__(self, key):
        i = hash(key) % self.blk
        del self.dicts[i][key]

    def has_key(self, key):
        i = hash(key) % self.blk
        return self.dicts[i].has_key(key)


    def __iter__(self):
        for i in self.dicts:
            for j in i:
                yield j

    def __len__(self):
        return sum([elem.size for elem in self.dicts])

# count 1 of the binary number
#def nbit(n):
#    x = n - ((n >> 1) & 033333333333) - ((n >> 2) & 011111111111)
#    return ((x + (x >> 3)) & 030707070707) % 63

def nbit(n):
    x = n - ((n >> 1) & 3681400539) - ((n >> 2) & 1227133513)
    return ((x + (x >> 3)) & 3340530119) % 63

nbit_jit_ = nb.njit(nbit)

# the last char of the kmer
# A: 1
# T: 10
# G: 100
# C: 1000
# N: 10000
# $: 100000
lastc = np.zeros(256, dtype='int8')
lastc[ord('a')] = lastc[ord('A')] = 0b1
lastc[ord('t')] = lastc[ord('T')] = 0b10
lastc[ord('g')] = lastc[ord('G')] = 0b100
lastc[ord('c')] = lastc[ord('C')] = 0b1000
lastc[ord('n')] = lastc[ord('N')] = 0b10000
lastc[ord('$')] = 0b100000 # end of the sequence
lastc[ord('#')] = 0b000000

offbit = int(math.log(max(lastc), 2)) + 1
#print('offbit', offbit, bin(max(lastc)))
lowbit = int('0b' + '1'* offbit, 2)
#offbit = 6
#print('offbit', offbit, 'low bin', bin(lowbit))

# reverse next character table
lastc_r = ['#'] * 0b100001
lastc_r[0b1] = 'A'
lastc_r[0b10] = 'T'
lastc_r[0b100] = 'G'
lastc_r[0b1000] = 'C'
lastc_r[0b10000] = 'N'
lastc_r[0b100000] = '$'
lastc_r = ''.join(lastc_r)


# convert dna kmer to number
# a:00, t:11, g:01, c:10
#alpha = array('i', [0] * 256)
alpha = np.zeros(256, dtype='int8')
alpha[:] = 0b100
alpha[ord('a')] = alpha[ord('A')] = 0b00
alpha[ord('t')] = alpha[ord('T')] = 0b11
alpha[ord('g')] = alpha[ord('G')] = 0b01
alpha[ord('c')] = alpha[ord('C')] = 0b10


# convert kmer to int
# bit is the length for encode atgc to number, default is 3
def k2n_(kmer, bit=5):
    N = 0
    for i in xrange(len(kmer)):
        c = alpha[ord(kmer[i])]
        N += c * bit ** i
    return N


beta = 'AGCTNNNN'
# convert int to kmer
# K is the length of kmer
def n2k_(N, K=12, bit=5):
    n, s = int(N), []
    for i in xrange(K):
        c = beta[n % bit]
        n //= bit
        s.append(c)
    return ''.join(s)

beta_ = np.frombuffer(beta.encode(), dtype='uint8')
@nb.njit
def n2k_jit_(N, K=12, bit=5, beta_=beta_):
    n, s = N, np.empty(K, dtype=nb.uint8)
    for i in xrange(K):
        #c = beta[n % bit]
        j = nb.uint64(n) % bit
        s[i] = beta_[j]
        n //= bit
    #return ''.join(s)
    return s



def seq2ns(seq, k=12):
    n = len(seq)
    if n < k:
        return -1

    Ns = [0] * (n-k+1)
    Ns[0] = flag = k2n_(seq[:k])
    shift = k*2-2
    for i in xrange(k, n):
        c = alpha[ord(seq[i])]
        flag = ((flag >> 2) | (c << shift))
        Ns[i-k+1] = flag

    return Ns

def seq2ns_0(seq, k=12, bit=5):
    n = len(seq)
    if n < k:
        #yield -1, '0', '0'
        yield -1, '#', '$'

    Nu = k2n_(seq[:k])
    #yield Nu, '0', seq[k]
    print('len', n, 'kmer', k)
    yield Nu, '#', seq[k]

    shift = bit ** (k - 1)
    for i in xrange(k, n-1):
        cc = alpha[ord(seq[i])]
        Nu = Nu // bit + cc * shift
        # find head and next char
        hd = seq[i-k]
        nc = seq[i+1]
        yield Nu, hd, nc

    cc = alpha[ord(seq[i+1])]
    Nu = Nu // bit + cc * shift
    hd = seq[i-k]
    yield Nu, hd, '$'


def seq2ns_0(seq, k=12, bit=5):
    n = len(seq)
    if n > k:
        Nu = k2n_(seq[:k])
        #yield Nu, '0', seq[k]
        #print('len', n, 'kmer', k)
        yield Nu, '#', seq[k]

        shift = bit ** (k - 1)
        for i in xrange(k, n-1):
            cc = alpha[ord(seq[i])]
            Nu = Nu // bit + cc * shift
            # find head and next char
            hd = seq[i-k]
            nc = seq[i+1]
            yield Nu, hd, nc

        cc = alpha[ord(seq[i+1])]
        Nu = Nu // bit + cc * shift
        hd = seq[i-k]
        yield Nu, hd, '$'

    elif n == k:
        #yield -1, '0', '0'
        yield k2n_(seq), '#', '$'
    else:
        yield -1, '#', '$'



def seq2ns_(seq, k=12, bit=5):
    n = len(seq)
    if n > k:
        Nu = k2n_(seq[:k])
        #yield Nu, '0', seq[k]
        #print('len', n, 'kmer', k)
        idx = 0
        yield idx, Nu, '#', seq[k]
        idx += 1

        shift = bit ** (k - 1)
        for i in xrange(k, n-1):
            cc = alpha[ord(seq[i])]
            Nu = Nu // bit + cc * shift
            # find head and next char
            hd = seq[i-k]
            nc = seq[i+1]
            yield idx, Nu, hd, nc
            idx += 1

        cc = alpha[ord(seq[i+1])]
        Nu = Nu // bit + cc * shift
        hd = seq[i-k]
        yield idx, Nu, hd, '$'

    elif n == k:
        #yield -1, '0', '0'
        yield 0, k2n_(seq), '#', '$'
    else:
        yield 0, -1, '#', '$'

# bisect based query
# xs is the sorted array
# x is the query
def query0(xs, x):
    idx = bisect_left(xs, x)
    return x in xs[idx:idx+1]

# check how many indegree and outdegree a node has
def query(ht, i):
    #print('breakpoint 582', ht.get([i]), i)
    try:
        hn = ht.get([i])
    except:
        return False

    if hn > 0:
        pr = nbit(hn >> offbit)
        sf = nbit(hn & lowbit)
        #print('pr', pr, 'sf', sf)
        if pr == sf == 1 and sf != 0b100000:
            return False
        else:
            return True
    else:
        return False


# check sequence's type
def seq_chk(qry):
    f = open(qry, 'r')
    seq = f.read(2*20)
    f.close()

    if seq[0].startswith('>') or '\n>' in seq:
        seq_type = 'fasta'
    elif seq[0].startswith('@') or '\n@' in seq:
        seq_type = 'fastq'
    else:
        seq_type = None

    return seq_type


# parse a file in the fasta/fastq format
def seqio(fn):
    # determine the sequences type
    seq_type = seq_chk(fn) 

    f = open(fn, 'r')
    if seq_type == 'fasta':
        head, seq = '', []
        for i in f:
            i = i.strip()
            if i.startswith('>'):
                if len(seq) > 0:
                    yield head, ''.join(seq)
                head = i[1:]
                seq = []
            else:
                seq.append(i)

        if len(seq) > 0:
            yield head, ''.join(seq)

    elif seq_type == 'fastq':
        head, seq = '', ''
        flag = 0
        for i in f:
            i = i.strip()
            if i.startswith('@'):
                head = i
                flag = 1
            elif flag == 1:
                seq = i
                yield head, seq
                flag = 0
            else:
                continue
        #if seq:
        #    yield head, ''.join(seq)

    else:
        raise SystemExit()

    f.close()


# reverse the sequence
tab_rev = {'A': 'T', 'a': 'T', 'T':'A', 't': 'A', 'G':'C', 'g': 'C', 'C':'G', 'c':'G'}
def reverse(seq, tab_rev=tab_rev):
    seq_rv = [tab_rev.get(elem, 'N') for elem in seq]
    seq_rv.reverse()
    return ''.join(seq_rv)

#tab_rev = ['N']*256
#for i, j in zip('atgc', 'TACG'):
#    tab_rev[ord(i)] = j
#    tab_rev[ord(i.upper())] = j
#def reverse(seq, tab_rev=tab_rev):
#    seq_rv = [tab_rev[ord(elem)] for elem in seq]
#    seq_rv.reverse()
#    return ''.join(seq_rv)

# function to compress genome sequence
def seq2dbg0(qry, kmer=13, bits=5, Ns=1e6):
    kmer = min(max(1, kmer), 27)
    size = int(pow(bits, kmer)+1)
    if kmer <= 13:
        #kmer_dict, fp = memmap('tmp.npy', shape=size, dtype='int16')
        kmer_dict = mmapht(size, 'int16')
    else:
        #kmer_dict = oaht(2**20, load_factor=.75)
        kmer_dict = mdict(2**20, load_factor=.75)
    #kmer_dict = robin(2**20, load_factor=.85)
   
    #for i in xrange(size):
    #    kmer_dict[i] = i

    N = 0
    for i in SeqIO.parse(qry, 'fasta'):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)

    #for header, sq in seqio(qry):
    #    seq_fw = sq
    #    seq_rv = reverse(sq)
    #    #print('seq', seq_fw[:10], seq_rv[:10])

        #itr = 0
        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                #if k == -1:
                #    continue
                #km = n2k_(k, kmer)
                #if km =='CCCC':
                #    print(km, hd, nt)

                h = lastc[ord(hd)] << offbit
                d = lastc[ord(nt)]
                try:
                    kmer_dict[k] |= (h | d) 
                except:
                    kmer_dict[k] = (h | d)

                #itr += 1
                #if itr % 10 ** 5 == 0:
                #    print('iter', itr)

            N += n
        print('N is', N)
        if N > Ns:
            break

    # dbg only store the branch
    #dbg = set()
    Graph = nx.DiGraph()

    dbg = []
    out_deg = 0
    nodes = 0
    #for i in xrange(len(kmer_dict)):
    for i in kmer_dict:
        #print('iter', i, len(kmer_dict), kmer_dict[i])
        try:
            hn = kmer_dict[i]
        except:
            print('not found', i, np.where(kmer_dict.keys == i))
            raise SystemExit()

        #if kmer_dict[i] > 0:
        if hn > 0:
            #print('i is', i, 'hn', hn, 'kmer', kmer)
            km = n2k_(int(i), kmer)
            #hn = kmer_dict[i]
            pr = nbit(hn >> offbit)
            sf = nbit(hn & lowbit)

            #if km == 'CCCC':
            #    print('kmer raw', km, bin(hn))

            if pr == sf == 1 and sf != 0b100000:
                #dbg.add(i)
                #dbg.append(i)
                #print('kmer', km, sf)
                pass
            else:
                dbg.append(i)
                #print('kmer', km, sf)

            #print('%s\t%s\t%s'%(km, pr, sf))
            out_deg += (sf > 1)
            nodes += 1

    print('dct size', len(kmer_dict), 'seq', N, 'nodes', nodes, 'branch', out_deg, 'rate', out_deg*100./N)
    #print('dbg', len(dbg), dbg)
    print('dbg', len(dbg))
    print('dbg', [n2k_(elem, kmer) for elem in dbg[:10]])

    dbg.sort()
    #raise SystemExit()
    N = 0

    for i in SeqIO.parse(qry, 'fasta'):
        seq_fw = str(i.seq)
 
    #for header, sq in seqio(qry, 'fasta'):
    #    seq_fw = sq

        path = []
        #print('seq', seq_fw)
        #for seq in [seq_fw, seq_rv]:
        for seq in [seq_fw]:
            skip = p0 = p1 = 0
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                if k == -1:
                    continue
                #idx = bisect_left(dbg, k)
                #print('kmer', k, idx)
                #if 0 < idx < len(dbg) or (idx == 0 and k == dbg[0]):
                #    path.append(idx)
                #else:
                #    print('not found', k)
                #    continue
                #path.append(k in dbg and k or -1)
                if p0 > p1:
                    p1 += 1
                    continue

                #if k in dbg:
                if query(dbg, k):
                    #if p0 <= p1:
                    #    path.append([skip, hd, k, k])
                    #    p0 += kmer
                    path.append(['p0', p0, 'p1', p1, 'skip', skip, hd, k, k, n2k_(k, K=kmer)])
                    p0 += kmer + skip
                    skip = 0
                else:
                    #path.append([hd, k, -1])
                    skip += 1
                p1 += 1
 
        for ii in xrange(len(path)-1):
            n0, n1 = path[ii:ii+2]
            #Graph.add_edge(n0[2], n1[2])
            #print('edge\t%d\t%d'%(n0[2], n1[2]))

        print('>' + i.id)
        #print(i.seq)
        print(path[:6])
        n = len(seq_fw)
        print('path', len(path), 'seq', len(seq_fw), n)
        N += n
        if N > Ns:
            break
    print('Graph size', Graph.size(), 'edge', len(Graph.edges()), 'node', len(Graph.nodes()))
    #for n0, n1 in Graph.edges():
    #    print('edge\t%d\t%d'%(n0, n1))
    return kmer_dict

def seq2dbg1(qry, kmer=13, bits=5, Ns=1e6):
    kmer = min(max(1, kmer), 27)
    size = int(pow(bits, kmer)+1)
    if kmer <= 13:
        kmer_dict = mmapht(size, 'int16')
    else:
        kmer_dict = oaht(2**20, load_factor=.75)

    N = 0

    f = open(qry, 'r')
    seq = f.read(10**6)
    f.close()

    if seq[0].startswith('>') or '\n>' in seq:
        seq_type = 'fasta'
    elif seq[0].startswith('@') or '\n@' in seq:
        seq_type = 'fastq'
    else:
        seq_type = None

    #print('input sequence is', seq_type, seq[:100])

    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)
 
    #for header, sq in seqio(qry):
    #    seq_fw = sq
    #    seq_rv = reverse(sq)
 
        for seq in [seq_fw, seq_rv][:1]:
            n = len(seq)
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                h = lastc[ord(hd)] << offbit
                d = lastc[ord(nt)]
                try:
                    kmer_dict[k] |= (h | d) 
                except:
                    kmer_dict[k] = (h | d)
            N += n
        #print('N is', N)
        if N > Ns:
            break

    # get frequency
    #for i in kmer_dict:
    #    print('freq', n2k_(i, kmer), kmer_dict.get_count(i))

    N = 0
    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)

    #for header, sq in seqio(qry, seq_type):
    #    seq_fw = sq

        path = []
        for seq in [seq_fw]:
            skip = p0 = p1 = 0
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                if k == -1:
                    continue
                if p0 > p1:
                    p1 += 1
                    continue

                if query(kmer_dict, k):
                    path.append(['p0', p0, 'p1', p1, 'skip', skip, hd, k, k, n2k_(k, K=kmer)])
                    p0 += kmer + skip
                    skip = 0
                else:
                    skip += 1
                p1 += 1

        #print('path', path)
        for ii in xrange(len(path)-1):
            n0, n1 = path[ii:ii+2]
            print('edge %s %s'%(n0[9], n1[9]))

        #print('>' + i.id)
        #print(i.seq)
        #print(path[:6])
        n = len(seq_fw)
        #print('path', len(path), 'seq', len(seq_fw), n)
        N += n
        if N > Ns:
            break

    return kmer_dict

# get the  breakpoint
def rec_bkt(f, seq_type):
    N = f.tell()
    header = seq_type == 'fastq' and '\n@' or '\n>'
    while 1:
        f.seek(N)
        if f.read(2) == header:
            N += 1
            #break
            return N
        else:
            N -= 1
    return 0

# adding resume function
def seq2dbg2(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**32, dump='breakpoint', saved='dBG_disk', hashfunc=oakht):
    kmer = min(max(1, kmer), 27)
    size = int(pow(bits, kmer)+1)
    breakpoint = 0
    if os.path.isfile(rec+'_seqs.txt'):
        #breakpoint, kmer_dict =resume[:2]
        f = open(rec + '_seqs.txt', 'r')
        breakpoint = int(f.next())
        f.close()
        #kmer_dict = oaht(2**20, load_factor=.75)
        #kmer_dict = oamkht(2**20, load_factor=.75)
        kmer_dict = hashfunc(2**30, load_factor=.75)

        print('rec is', rec, kmer_dict)
        kmer_dict.loading(rec)
        print('the size oaht', len(kmer_dict))

    elif kmer <= 13:
        kmer_dict = mmapht(size, 'int16')
    else:
        #kmer_dict = oaht(2**20, load_factor=.75)
        #kmer_dict = oamkht(2**20, load_factor=.75)
        kmer_dict = hashfunc(2**30, load_factor=.75)

    N = 0

    f = open(qry, 'r')
    seq = f.read(10**6)
    f.close()

    if seq[0].startswith('>') or '\n>' in seq:
        seq_type = 'fasta'
    elif seq[0].startswith('@') or '\n@' in seq:
        seq_type = 'fastq'
    else:
        seq_type = None

    print('breakpoint is', breakpoint)
    #print('input sequence is', seq_type, seq[:100])
    f = open(qry, 'r')
    f.seek(breakpoint)
    flag = 0

    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)

    #for header, sq in seqio(qry):
    #    seq_fw = sq
    #    seq_rv = reverse(sq)
 
        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                h = lastc[ord(hd)] << offbit
                d = lastc[ord(nt)]
                try:
                    kmer_dict[k] |= (h | d) 
                except:
                    kmer_dict[k] = (h | d)
            N += n
            flag += n

        # save the breakpoint
        if flag > chunk:
            bkt = rec_bkt(f, seq_type)
            print('breakpoint', bkt)
            _o = open(dump+'_seqs.txt', 'w')
            _o.write('%d'%bkt)
            _o.close()
            flag = 0
            kmer_dict.dump(dump)

        #print('N is', N)
        if N > Ns:
            break

    # save the de bruijn graph to disk
    kmer_dict.dump(saved)

    f.close()
    # get frequency
    for i in kmer_dict:
        print('key', i)
        print('freq', n2k_(i, kmer))
        print('#'*10)
        print('size', len(kmer_dict), 'freq', n2k_(i, kmer), kmer_dict.get_count(i))
        break

    N = 0
    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)
    #for header, sq in seqio(qry):
    #    seq_fw = sq

        path = []
        for seq in [seq_fw]:
            skip = p0 = p1 = 0
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                if k == -1:
                    continue
                if p0 > p1:
                    p1 += 1
                    continue

                if query(kmer_dict, k):
                    path.append(['p0', p0, 'p1', p1, 'skip', skip, hd, k, k, n2k_(k, K=kmer)])
                    p0 += kmer + skip
                    skip = 0
                else:
                    skip += 1
                p1 += 1

            # if path is empty, then sequence has non-repetitive k-mer, add the last k-mer to path
            if not path:
                path = ['p0', p0, 'p1', p1, 'skip', skip, hd, k, k, n2k_(k, K=kmer)]


        print(seq_fw[:27])
        print('path', path)
        #for ii in xrange(len(path)-1):
        #    n0, n1 = path[ii:ii+2]
        #    print('edge %s %s'%(n0[9], n1[9]))

        #print('>' + i.id)
        #print(i.seq)
        #print(path[:6])
        n = len(seq_fw)
        #print('path', len(path), 'seq', len(seq_fw), n)
        N += n
        if N > Ns:
            break

    return kmer_dict


# find weight for rDBG
def seq2dbg(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**32, dump='breakpoint', saved='dBG_disk', hashfunc=oakht):
    kmer = min(max(1, kmer), 27)
    size = int(pow(bits, kmer)+1)
    breakpoint = 0
    if os.path.isfile(rec+'_seqs.txt'):
        #breakpoint, kmer_dict =resume[:2]
        f = open(rec + '_seqs.txt', 'r')
        breakpoint = int(f.next())
        f.close()
        kmer_dict = hashfunc(2**20, load_factor=.75)

        print('rec is', rec, kmer_dict)
        kmer_dict.loading(rec)
        print('the size oaht', len(kmer_dict))

    elif kmer <= 13:
        kmer_dict = mmapht(size, 'int16')
    else:
        kmer_dict = hashfunc(2**20, load_factor=.75)

    N = 0
    f = open(qry, 'r')
    seq = f.read(10**6)
    f.close()

    if seq[0].startswith('>') or '\n>' in seq:
        seq_type = 'fasta'
    elif seq[0].startswith('@') or '\n@' in seq:
        seq_type = 'fastq'
    else:
        seq_type = None

    print('breakpoint is', breakpoint)
    f = open(qry, 'r')
    f.seek(breakpoint)
    flag = 0

    for i in SeqIO.parse(f, seq_type):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)

    #for header, sq in seqio(qry):
    #    seq_fw = sq
    #    seq_rv = reverse(sq)
 
        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                h = lastc[ord(hd)] << offbit
                d = lastc[ord(nt)]
                try:
                    kmer_dict[k] |= (h | d) 
                except:
                    kmer_dict[k] = (h | d)
            N += n
            flag += n

        # save the breakpoint
        if flag > chunk:
            bkt = rec_bkt(f, seq_type)
            print('breakpoint', bkt)
            _o = open(dump+'_seqs.txt', 'w')
            _o.write('%d'%bkt)
            _o.close()
            flag = 0
            kmer_dict.dump(dump)

        #print('N is', N)
        if N > Ns:
            break

    # save the de bruijn graph to disk
    kmer_dict.dump(saved)

    f.close()

    # get frequency
    for i in kmer_dict:
        print('key', i)
        print('freq', n2k_(i, kmer))
        print('#'*10)
        print('size', len(kmer_dict), 'freq', n2k_(i, kmer), kmer_dict.get_count(i))
        break

    # find weight for rDBG
    rdbg = oamkht(mkey=2, val_type='uint32')

    N = 0

    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)

    #for header, sq in seqio(qry):                                                                                           
    #    seq_fw = sq

        for seq in [seq_fw]:
            path_cmpr = []
            path_rdbg = []
            skip = p0 = p1 = 0
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                if k == -1:
                    continue
                #if p0 > p1:
                #    p1 += 1
                #    #continue

                if query(kmer_dict, k):
                    path_rdbg.append(k)
                    if p0 <= p1:
                        path_cmpr.append(['p0', p0, 'p1', p1, 'skip', skip, hd, k, k, n2k_(k, K=kmer)])
                        p0 += kmer + skip
                        skip = 0
                else:
                    skip += 1
                p1 += 1

            # if path is empty, then sequence has non-repetitive k-mer, add the last k-mer to path
            if not path_cmpr:
                path_cmpr = [['p0', p0, 'p1', p1, 'skip', skip, hd, k, k, n2k_(k, K=kmer)]]

            print(seq_fw[:27])
            print('path', len(path_cmpr))
            for ii in xrange(len(path_rdbg)-1):
                n0, n1 = path_rdbg[ii:ii+2]
                #print('edge %s %s'%(n0[9], n1[9]))
                k12 = (n0, n1)
                try:
                    rdbg[k12] += 1
                except:
                    rdbg[k12] = 1

            #print('>' + i.id)
            #print(i.seq)
            #print(path[:6])
            n = len(seq)
            #print('path', len(path), 'seq', len(seq_fw), n)
        N += n
        if N > Ns:
            break

    print('rdbg size', len(rdbg))
    for k12 in rdbg:
        n0, n1 = k12
        print('edge', n0, n1, rdbg[k12])
    return kmer_dict


# load dbg on disk
def load_dbg(saved, kmer_dict):

    dump_keys, dump_fk = memmap(saved + '_dump_key.npy', 'a+', dtype='uint64')
    dump_values, dump_fv = memmap(saved + '_dump_val.npy', 'a+', dtype='uint16')
    dump_counts, dump_fc = memmap(saved + '_dump_cnt.npy', 'a+', dtype='uint8')

    for i in xrange(dump_values.shape[0]):
        if query(dump_values, i):
            key, val = dump_keys[i], dump_values[i]
            kmer_dict[key] = val
        #else:
        #    print('dropped', i)
    # print('rdbg size', len(kmer_dict))

    dump_fk.close()
    dump_fv.close()
    dump_fc.close()

    #return kmer_dict

# convert sequences to rdbg
def seq2rdbg0(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**32, dump='breakpoint', saved='dBG_disk', hashfunc=oakht):
    kmer = min(max(1, kmer), 27)
    size = int(pow(bits, kmer)+1)
    breakpoint = 0
    if os.path.isfile(rec+'_seqs.txt'):
        #breakpoint, kmer_dict =resume[:2]
        f = open(rec + '_seqs.txt', 'r')
        breakpoint = int(f.next())
        f.close()
        kmer_dict = hashfunc(2**20, load_factor=.75)
        #print('rec is', rec, kmer_dict)
        kmer_dict.loading(rec)
        #print('the size oaht', len(kmer_dict))
        #print('wocaonima')

    elif kmer <= 13:
        kmer_dict = mmapht(size, 'int16')
        #print('wocaonima')

    else:
        kmer_dict = hashfunc(2**20, load_factor=.75)
  
    print('rdbg size', len(kmer_dict))

    #seq_type = seq_chk(qry)

    N = 0
    print('breakpoint is', breakpoint)
    f = open(qry, 'r')
    f.seek(breakpoint)
    flag = 0

    for i in SeqIO.parse(f, seq_type):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)

    #for header, sq in seqio(qry):                                                                                           
    #    seq_fw = sq
    #    seq_rv = reverse(sq)

        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                h = lastc[ord(hd)] << offbit
                d = lastc[ord(nt)]
                try:
                    kmer_dict[k] |= (h | d) 
                except:
                    kmer_dict[k] = (h | d)
            N += n
            flag += n

        # save the breakpoint
        if flag > chunk:
            bkt = rec_bkt(f, seq_type)
            print('breakpoint', bkt)
            _o = open(dump+'_seqs.txt', 'w')
            _o.write('%d'%bkt)
            _o.close()
            flag = 0
            kmer_dict.dump(dump)

        #print('N is', N)
        if N > Ns:
            break

    # save the de bruijn graph to disk
    kmer_dict.dump(saved)

    f.close()
    return kmer_dict


@nb.njit(inline='always')
def k2n_jit_(kmer, bit=5, alpha=alpha):
    #N = nb.ulonglong(0)
    N = idx = 0
    #for i in xrange(len(kmer)):
    #idx = 0
    for i in kmer:
        c = nb.ulonglong(alpha[i])
        N += c * bit ** idx
        idx += 1

    return N

@nb.njit
def seq2ns_jit_(seq, k=12, bit=5, alpha=alpha):
    # '#' is 35, '$' is 36
    output = np.empty(4, dtype=np.uint64)
    n = len(seq)
    if n > k:
        Nu = k2n_jit_(seq[:k])
        idx = 0
        #output[0:4] = idx, Nu, 35, seq[k]
        output[0], output[1], output[2], output[3] = idx, Nu, 35, seq[k]
        yield output
        idx += 1

        shift = bit ** (k - 1)
        for i in xrange(k, n-1):
            cc = alpha[seq[i]]
            Nu = Nu // bit + cc * shift
            # find head and next char
            hd = int(seq[i-k])
            nc = int(seq[i+1])
            #yield idx, Nu, hd, nc
            output[0], output[1], output[2], output[3] = idx, Nu, hd, nc
            yield output
            idx += 1

        #cc = alpha[ord(seq[i+1])]
        cc = alpha[seq[i+1]]
        Nu = Nu // bit + cc * shift
        hd = int(seq[i-k])
        #yield idx, Nu, hd, end
        output[0], output[1], output[2], output[3] = idx, Nu, hd, 36
        yield output

    elif n == k:
        #yield 0, k2n_jit_(seq), 35, 36
        output[0], output[1], output[2], output[3] = 0, k2n_jit_(seq), 35, 36
        yield output

    else:
        #yield 0, -1, 35, 36
        #output[0], output[1], output[2], output[3] = 0, -1, 35, 36
        output[0], output[1], output[2], output[3] = 0, -1, 35, 36
        yield output

@nb.njit(inline='always')
def add_kmer(kmer_dict, key, idx, Nu, hd, nc, lastc=lastc, offbit=offbit):
    key[0], hd, nt = Nu, hd, nc
    h = lastc[hd] << offbit
    d = lastc[nt]
    if kmer_dict.has_key(key):
        val = kmer_dict.get(key) 
        kmer_dict.push(key, val | h | d)
    else:
        kmer_dict.push(key, h | d)

@nb.njit
def kmer2dict(seq, kmer_dict, kmer=12, bit=5, offbit=offbit, lastc=lastc, alpha=alpha):
    k = kmer
    key = np.empty(1, dtype=np.uint64)
    # '#' is 35, '$' is 36
    #output = np.empty(4, dtype=np.uint64)
    n = len(seq)
    if n > k:
        Nu = k2n_jit_(seq[:k])
        idx = 0

        add_kmer(kmer_dict, key, idx, Nu, 35, seq[k], lastc=lastc, offbit=offbit)
        #idx, key[0], hd, nt = idx, Nu, 35, seq[k]
        #h = lastc[hd] << offbit
        #d = lastc[nt]
        #if kmer_dict.has_key(key):
        #    val = kmer_dict.get(key) 
        #    kmer_dict.push(key, val | h | d)
        #else:
        #    kmer_dict.push(key, h | d)

        #idx += 1
        shift = bit ** (k - 1)
        for i in xrange(k, n-1):
            cc = alpha[seq[i]]
            Nu = Nu // bit + cc * shift
            # find head and next char
            hd = int(seq[i-k])
            nc = int(seq[i+1])

            add_kmer(kmer_dict, key, idx, Nu, hd, nc, lastc=lastc, offbit=offbit)
            #idx, key[0], hd, nt = idx, Nu, hd, nc
            #h = lastc[hd] << offbit
            #d = lastc[nt]
            #if kmer_dict.has_key(key):
            #    val = kmer_dict.get(key) 
            #    kmer_dict.push(key, val | h | d)
            #else:
            #    kmer_dict.push(key, h | d)

            #idx += 1
            #if idx % 10**7 == 0:
            #    print('idx is', idx)

        #cc = alpha[ord(seq[i+1])]
        cc = alpha[seq[i+1]]
        Nu = Nu // bit + cc * shift
        hd = int(seq[i-k])
        #yield idx, Nu, hd, end
        #output[0], output[1], output[2], output[3] = idx, Nu, hd, 36
        #yield output

        add_kmer(kmer_dict, key, idx, Nu, hd, 36, lastc=lastc, offbit=offbit)
        #idx, key[0], hd, nt = idx, Nu, hd, 36
        #h = lastc[hd] << offbit
        #d = lastc[nt]
        #if kmer_dict.has_key(key):
        #    val = kmer_dict.get(key)
        #    print('yes, wtf before', val,'h', h, 'd', d, val|h|d)
        #    #kmer_dict.push(key, val|h|d)
        #    add_kmer(kmer_dict, key, idx, Nu, hd, 36, lastc=lastc, offbit=offbit)
        #    print('yes, wtf after', kmer_dict.get(key), val|h|d, kmer_dict.get(key)==val|h|d)
        #else:
        #    print('no, wtf before', val, 'h', h, 'd', d, h|d)
        #    kmer_dict.push(key, h|d)
        #    add_kmer(kmer_dict, key, idx, Nu, hd, 36, lastc=lastc, offbit=offbit)
        #    print('no, wtf after', kmer_dict.get(key), h|d, kmer_dict.get(key)==h|d)

    elif n == k:
        #yield 0, k2n_jit_(seq), 35, 36
        #output[0], output[1], output[2], output[3] = 0, k2n_jit_(seq), 35, 36
        #yield output

        add_kmer(kmer_dict, key, 0, k2n_jit_(seq), 35, 36, lastc=lastc, offbit=offbit)
        #idx, key[0], hd, nt = 0, k2n_jit_(seq), 35, 36
        #h = lastc[hd] << offbit
        #d = lastc[nt]
        #if kmer_dict.has_key(key):
        #    val = kmer_dict.get(key) 
        #    kmer_dict.push(key, val | h | d)
        #else:
        #    kmer_dict.push(key, h | d)

    else:
        #yield 0, -1, 35, 36
        #output[0], output[1], output[2], output[3] = 0, -1, 35, 36
        #output[0], output[1], output[2], output[3] = 0, -1, 35, 36
        #yield output

        add_kmer(kmer_dict, key, 0, -1, 35, 36, lastc=lastc, offbit=offbit)
        #idx, key[0], hd, nt = 0, -1, 35, 36
        #h = lastc[hd] << offbit
        #d = lastc[nt]
        #if kmer_dict.has_key(key):
        #    val = kmer_dict.get(key) 
        #    kmer_dict.push(key, val | h | d)
        #else:
        #    kmer_dict.push(key, h | d)
    return 0

spec = {}
spec['capacity'] = nb.int64
spec['load'] = nb.float32
spec['size'] = nb.int64
spec['ksize'] = nb.int64
spec['ktype'] = nb.uint64
spec['keys'] = spec['ktype'][:]
spec['vtype'] = nb.uint16
spec['values'] = spec['vtype'][:]
spec['counts'] = nb.uint8[:]

# init my own hash table in jit
def init_dict(hashfunc=oakht, capacity=2**20, ksize=1, ktype=nb.uint64, vtype=nb.uint32, jit=True):
    if jit:
        spec = {}
        spec['capacity'] = nb.int64
        spec['load'] = nb.float32
        spec['size'] = nb.int64
        spec['ksize'] = nb.int64

        #spec['ktype'] = nb.uint64
        #spec['keys'] = spec['ktype'][:]
        #spec['vtype'] = nb.uint16
        #spec['values'] = spec['vtype'][:]

        spec['ktype'] = ktype
        spec['keys'] = spec['ktype'][:]
        spec['vtype'] = vtype
        spec['values'] = spec['vtype'][:]

        spec['counts'] = nb.uint8[:]
        clf = nb.jitclass(spec)(hashfunc)

    else:
        clf = hashfunc

    dct = clf(capacity=2**20, ksize=1, ktype=nb.uint64, vtype=nb.uint16)
    return dct
 

# build the dBG
#def seq2rdbg(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**32, dump='breakpoint', saved='dBG_disk', hashfunc=oakht, jit=True, spec=spec):
def seq2rdbg(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**32, dump='breakpoint', saved='dBG_disk', hashfunc=oakht, jit=True):

    kmer = min(max(1, kmer), 27)
    size = int(pow(bits, kmer)+1)
    breakpoint = 0
    if os.path.isfile(rec+'_seqs.txt'):
        f = open(rec + '_seqs.txt', 'r')
        breakpoint = int(f.next())
        f.close()
        kmer_dict = hashfunc(2**20, load_factor=.75)
        kmer_dict.loading(rec)

    #elif kmer <= 13:
    #    kmer_dict = mmapht(size, 'int16')

    else:
        #if jit and spec:
        #    print('using jit')
        #    hashfunc_jit = nb.jitclass(spec)(hashfunc)
        #else:
        #    print('no jit')
        #    hashfunc_jit = hashfunc
        #hashfunc_jit = init_dict(hashfunc=oakht, capacity=2**20, ksize=1, ktype=nb.uint64, vtype=nb.uint16, jit=jit)

        #kmer_dict = hashfunc_jit(2**20, load_factor=.75)
        #kmer_dict = hashfunc_jit(capacity=2**20, ksize=1, ktype=spec['ktype'], vtype=spec['vtype'])
        #kmer_dict = hashfunc_jit(capacity=2**20, ksize=1, ktype=nb.uint64, vtype=nb.uint16)
        kmer_dict = init_dict(hashfunc=oakht, capacity=2**20, ksize=1, ktype=nb.uint64, vtype=nb.uint16, jit=jit)



    #print('rdbg size', len(kmer_dict))
    seq_type = seq_chk(qry)

    N = 0
    print('breakpoint is', breakpoint)
    f = open(qry, 'r')
    f.seek(breakpoint)
    flag = 0

    for i in SeqIO.parse(f, seq_type):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)

        print('before adding node', i.id, len(seq_fw), 'clf size', kmer_dict.size, 'seq len', N)

        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            #for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
            #    h = lastc[ord(hd)] << offbit
            #    d = lastc[ord(nt)]
            #    try:
            #        kmer_dict[k] |= (h | d) 
            #    except:
            #        kmer_dict[k] = (h | d)

            seq_array = np.frombuffer(seq.encode(), dtype='uint8')

            #kmer2dict(seq_array, kmer, bits, lastc, offbit, kmer_dict)
            res = kmer2dict(seq_array, kmer_dict, kmer=kmer, bit=bits, offbit=offbit, lastc=lastc, alpha=alpha)


            N += n
            flag += n

        print('after adding nodes', i.id, 'clf size', kmer_dict.size, 'seq len', N)

        # save the breakpoint
        if flag > chunk:
            bkt = rec_bkt(f, seq_type)
            print('breakpoint', bkt)
            _o = open(dump+'_seqs.txt', 'w')
            _o.write('%d'%bkt)
            _o.close()
            flag = 0
            kmer_dict.dump(dump)

        #print('N is', N)
        if N > Ns:
            break

    # save the de bruijn graph to disk
    #kmer_dict.dump(saved)

    f.close()

    #for hn in kmer_dict.itervalues():
    #    pr = nbit(hn >> offbit)
    #    sf = nbit(hn & lowbit)
    #    #print('pr', pr, 'sf', sf)
    #    if pr == sf == 1 and sf != 0b100000:
    #        continue
    #    else:
    #        print('1615 yes, find', pr, sf)
    #
    #print('1585 end')
    #raise SystemExit()

    return kmer_dict

# check if a kmer in dbg is also in rdbg:
@nb.njit
def kmer_in_rdbg(ht, i):
    if ht.has_key(i):
        hn = ht.get([i])
    else:
        return False

    if hn > 0:
        pr = nbit_jit_(hn >> offbit)
        sf = nbit_jit_(hn & lowbit)
        #print('pr', pr, 'sf', sf)
        if pr == sf == 1 and sf != 0b100000:
            return False
        else:
            return True
    else:
        return False


# build rdbg from dbg
@nb.njit
def build_rdbg(rdbg_dict, kmer_dict):
    for kv in kmer_dict.iteritems():
        k, hn = kv
        #print('kv is', type(k))
        pr = nbit_jit_(hn >> offbit)
        sf = nbit_jit_(hn & lowbit)
        if pr == sf == 1 and sf != 0b100000:
            continue
        else:
            rdbg_dict.push(k, hn)

    return rdbg_dict


# get the weight of edge
@nb.njit
def rdbg_edge_weight(rdbg_edge, rdbg_dict, seq, kmer, bits=5):
    minus_one = nb.uint64(-1)

    # path_cmpr has 4 elem
    #path_cmpr = List()
    #path_cmpr.append(minus_one)
    #path_cmpr.clear()
    path_cmpr = List.empty_list(nb.uint64)

    # path_rdbg has 3 elem
    #path_rdbg = List()
    #path_rdbg.append(minus_one)
    #path_rdbg.clear()
    path_rdbg = List.empty_list(nb.uint64)

   
    idx, idx_prev = minus_one, minus_one
    for output in seq2ns_jit_(seq, kmer, bits):
        #idx, k, hd, nt = output[0], output[1], output[2], output[3]
        idx, k, hd, nt = output
        if k == minus_one:
            continue

        key = output[1:2]

        if rdbg_dict.has_key(key):

            #path_rdbg.append((k, hd, nt))
            #path_rdbg.extend((k, hd, nt))
            path_rdbg.extend(output[1:4])
            #path_rdbg.append(output[1:4])
            #path_rdbg.append(k)
            #path_rdbg.append(hd)
            #path_rdbg.append(nt)

            if idx_prev == minus_one or idx_prev + kmer <= idx:
                #path_cmpr.append([idx, k, hd, nt, n2k_(k, K=kmer)])
                #path_cmpr.extend([idx, k, hd, nt])
                path_cmpr.extend(output)
                #path_cmpr.append(output[:])
                #path_cmpr.append(idx)
                #path_cmpr.append(k)
                #path_cmpr.append(hd)
                #path_cmpr.append(nt)
                idx_prev = idx

    # if path is empty, then sequence has non-repetitive k-mer, add the last k-mer to path
    if len(path_cmpr) == 0:
        #path_cmpr = [[idx, k, hd, nt, n2k_(k, K=kmer)]]
        #path_cmpr = [[idx, k, hd, nt]]
        #path_cmpr = [idx, k, hd, nt]
        path_cmpr.extend(output)
        #path_cmpr.append(output[:])
        #path_cmpr.append(idx)
        #path_cmpr.append(k)
        #path_cmpr.append(hd)
        #path_cmpr.append(nt)

    N = len(path_rdbg)
    visit = {}
    #for i in range(N-1):
    for i in range(0, N-3, 3):
        #n0, n1 = path_rdbg[idx:idx+2]
        #n0, hd0, nt0 = path_rdbg[idx]
        n0, hd0, nt0 = path_rdbg[i: i+3]
        #n0, hd0, nt0 = path_rdbg[idx], path_rdbg[idx+1], path_rdbg[idx+2]

        h0 = lastc[hd0] << offbit
        d0 = lastc[nt0]
 
        #n1, hd1, nt1 = path_rdbg[idx+1]
        n1, hd1, nt1 = path_rdbg[i+3: i+6]
        #n1, hd1, nt1 = path_rdbg[idx+3], path_rdbg[idx+4], path_rdbg[idx+5]

        h1 = lastc[hd1] << offbit
        d1 = lastc[nt1]

        k12 = (n0, nb.uint64(h0|d0), n1, nb.uint64(h1|d1))

        # remove the repeat in the same sequences
        if k12 not in visit:
            visit[k12] = 1
        else:
            continue

        try:
            rdbg_edge[k12] += 1
        except:
            rdbg_edge[k12] = 1

    print('1803 break rdbg_edge size', len(rdbg_edge))
    del visit
    return 0

# convert sequences to paths and build the graph
def seq2graph(qry, kmer=13, bits=5, Ns=1e6, kmer_dict=None, saved=None, hashfunc=oakht, jit=True, spec=spec):

    kmer = min(max(1, kmer), 27)
    if kmer_dict != None:
        saved = None
    elif kmer <= 13:
        size = int(pow(bits, kmer)+1)
        kmer_dict = mmapht(size, 'int16')
    else:
        kmer_dict = hashfunc(2**20, load_factor=.75)

    # find the type of sequences
    seq_type = seq_chk(qry)

    # load the graph on disk
    if saved:
        #print('before load', len(kmer_dict))
        load_dbg(saved, kmer_dict)

    #print('kmer_dict of size', len(kmer_dict), saved)
    # find weight for rDBG
    #rdbg = oamkht(mkey=4, val_type='uint32')
    #spec['vtype'] = nb.uint16
    #spec['values'] = spec['vtype'][:]
    #if jit and spec:
    #    hashfunc_jit = nb.jitclass(spec)(hashfunc)
    #else:
    #    hashfunc_jit = hashfunc

    # use numba dict
    #rdbg = hashfunc_jit(capacity=2**20, ksize=4, ktype=spec['ktype'], vtype=spec['vtype'])
    
    #hashfunc_jit = init_dict(hashfunc=oakht, ktype=nb.uint64, vtype=nb.uint16, jit=jit)

    rdbg_dict = init_dict(hashfunc=oakht, capacity=2**20, ksize=kmer_dict.ksize, ktype=nb.uint64, vtype=nb.uint16, jit=jit)
    rdbg_dict = build_rdbg(rdbg_dict, kmer_dict)


    del kmer_dict
    gc.collect()

    rdbg_edge = Dict()
    zero = nb.uint64(0)
    rdbg_edge[(zero, zero, zero, zero)] = 0
    #rdbg_edge[(0, 0, 0, 0)] = 0
    N = 0

    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)
        for seq in [seq_fw]:

            seq_array = np.frombuffer(seq.encode(), dtype='uint8')
            res = rdbg_edge_weight(rdbg_edge, rdbg_dict, seq_array, kmer, bits)
            #print('1856 break', len(rdbg_edge))
            #path_cmpr = []
            #path_rdbg = []
            #idx_prev = -1
            #for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
            #    if k == -1:
            #        continue
            #
            #    if query(kmer_dict, k):
            #        path_rdbg.append([k, hd, nt])
            #        if idx_prev == -1 or idx_prev + kmer <= idx:
            #            path_cmpr.append([idx, k, hd, nt, n2k_(k, K=kmer)])
            #            idx_prev = idx
            #
            ## if path is empty, then sequence has non-repetitive k-mer, add the last k-mer to path
            #if not path_cmpr:
            #    path_cmpr = [[idx, k, hd, nt, n2k_(k, K=kmer)]]
            #
            #print('path 1635', path_rdbg)
            #visit = set()
            #for idx in xrange(len(path_rdbg)-1):
            #    #n0, n1 = path_rdbg[idx:idx+2]
            #    n0, hd0, nt0 = path_rdbg[idx]
            #    h0 = lastc[ord(hd0)] << offbit
            #    d0 = lastc[ord(nt0)]
            #
            #    n1, hd1, nt1 = path_rdbg[idx+1]
            #    h1 = lastc[ord(hd1)] << offbit
            #    d1 = lastc[ord(nt1)]
            #
            #    k12 = np.asarray([n0, h0|d0, n1, h1|d1])
            #
            #    # remove the repeat in the same sequences
            #    if k12 not in visit:
            #        visit.add(k12)
            #    else:
            #        continue
            #
            #    try:
            #        rdbg[k12] += 1
            #    except:
            #        rdbg[k12] = 1

            n = len(seq)
        N += n
        if N > Ns:
            break

    print('1902 break end', len(rdbg_edge))
    raise SystemExit()

    print('rdbg size 1665', len(rdbg))
    _oname = qry + '_rdbg_weight.xyz'
    _o = open(_oname, 'w')
    for k12 in rdbg.iteritems():
        #n0, n1 = k12
        (n0, hd0, n1, hd1), val = k12
        #xyz = '%d\t%d\t%d\n'%(n0, n1, rdbg[k12])
        #xyz = '%d_%d\t%d_%d\t%d\n'%(n0, hd0, n1, hd1, rdbg[k12])
        xyz = '%d_%d\t%d_%d\t%d\n'%(n0, hd0, n1, hd1, val)
        _o.write(xyz)

    _o.close()

    # call the mcl for clustering
    os.system('mcl %s --abc -I 1.5 -te 8 -o %s.mcl'%(_oname, _oname))

    del rdbg
    del kmer_dict
    gc.collect()

    # load the cluster
    #label_dct = oamkht(2 ** 20, val_type='int32')
    #label_dct = oamkht(2 ** 20, mkey=2, val_type='int32')

    spec['vtype'] = nb.uint32
    spec['values'] = spec['vtype'][:]
    if jit and spec:
        hashfunc_jit = nb.jitclass(spec)(hashfunc)
    else:
        hashfunc_jit = hashfunc
 
    label_dct = hashfunc_jit(capacity=2**20, ksize=2, ktype=spec['ktype'], vtype=spec['vtype'])

    flag = 0
    f = open(_oname+'.mcl', 'r')
    for i in f:
        j = i[:-1].split('\t')
        for k in j:
            ky = tuple(map(int, k.split('_')[:2]))
            #label_dct[int(k)] = flag
            label_dct[ky] = flag

        flag += 1

    f.close()


    # add the rest kmer
    f = open(_oname, 'r')
    for i in f:
        j, k = i[:-1].split('\t')[:2]

        kj = tuple(map(int, j.split('_')[:2]))
        #if not label_dct.has_key(j):
        if not label_dct.has_key(kj):
            #label_dct[j] = flag
            label_dct[kj] = flag
            flag += 1

        kk = tuple(map(int, k.split('_')[:2]))
        #if not label_dct.has_key(j):
        if not label_dct.has_key(kk):
            #label_dct[k] = flag
            label_dct[kk] = flag
            flag += 1


    N = 0
    print('label_dct', label_dct.size)
    for i in SeqIO.parse(qry, seq_type):
    #    #print('id', i.id)
        seq_fw = str(i.seq)

    #for header, sq in seqio(qry):                                                                                           
    #    seq_fw = sq

        for seq in [seq_fw]:
            starts = [0]
            labels = [-1]
            #idx_pre, label_pre = 0, -1
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):

                h = lastc[ord(hd)] << offbit
                d = lastc[ord(nt)]
                kk = (k, h|d)
 
                #if label_dct.has_key(k):
                if label_dct.has_key(kk):
                    #label = label_dct[k]
                    label = label_dct[kk]
                    if starts[-1] < idx:

                        # find different conserved region
                        if labels[-1] != label:
                            labels.append(label)
                            starts.append(idx+kmer)
                        # the same region, just extend it.
                        else:
                            starts[-1] = idx + kmer

                    #if idx_pre + kmer > idx:
                    #if starts[-1] == 0:
                    #    if starts[-1] != idx:
                    #        #starts.append(idx)
                    #        starts.append(idx+kmer)
                    #        labels.append(label)

                    #elif starts[-1] + kmer > idx:
                    #elif starts[-1] > idx:
                    #if starts[-1] > idx:
                    #    continue

                    #elif labels[-1] != label:
                    #    labels.append(label)
                    #    #starts.append(idx)
                    #    starts.append(idx+kmer)

                    #else:
                    #    #starts[-1] = idx
                    #    starts[-1] = idx + kmer
                    #    #idx_pre = idx

            #print('starts', len(starts), starts[:5], labels[:5], len(seq))
            #print('%s\t%d\t%d\t%s\t%d'%(i.id, 0, starts[0], '+', labels[0]))
            for idx in xrange(1, len(starts)):
                #print('%s\t%d\t%d\t%s\t%d'%(header.split(' ')[0], starts[idx-1], starts[idx], '+', labels[idx]))
                print('%s\t%d\t%d\t%s\t%d'%(i.id, starts[idx-1], starts[idx], '+', labels[idx]))
                #print('%s\t%d\t%d\%s\%d'%(i.id, start, end, '+', label))

        N += len(seq_fw)
        if N > Ns:
            break

    print('end', 1798)
    #return kmer_dict
    return label_dct




def seq2graph0(qry, kmer=13, bits=5, Ns=1e6, kmer_dict=None, saved=None, hashfunc=oakht, jit=True, spec=spec):

    kmer = min(max(1, kmer), 27)
    if kmer_dict != None:
        saved = None
    elif kmer <= 13:
        size = int(pow(bits, kmer)+1)
        kmer_dict = mmapht(size, 'int16')
    else:
        kmer_dict = hashfunc(2**20, load_factor=.75)

    # find the type of sequences
    seq_type = seq_chk(qry)

    # load the graph on disk
    if saved:
        #print('before load', len(kmer_dict))
        load_dbg(saved, kmer_dict)

    spec['vtype'] = nb.uint16
    spec['values'] = spec['vtype'][:]
    if jit and spec:
        hashfunc_jit = nb.jitclass(spec)(hashfunc)
    else:
        hashfunc_jit = hashfunc
    rdbg = hashfunc_jit(capacity=2**20, ksize=4, ktype=spec['ktype'], vtype=spec['vtype'])

    N = 0

    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)

        for seq in [seq_fw]:
            path_cmpr = []
            path_rdbg = []
            idx_prev = -1
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
                if k == -1:
                    continue

                if query(kmer_dict, k):
                    path_rdbg.append([k, hd, nt])
                    if idx_prev == -1 or idx_prev + kmer <= idx:
                        path_cmpr.append([idx, k, hd, nt, n2k_(k, K=kmer)])
                        idx_prev = idx

            # if path is empty, then sequence has non-repetitive k-mer, add the last k-mer to path
            if not path_cmpr:
                path_cmpr = [[idx, k, hd, nt, n2k_(k, K=kmer)]]

            visit = set()
            for idx in xrange(len(path_rdbg)-1):
                n0, hd0, nt0 = path_rdbg[idx]
                h0 = lastc[ord(hd0)] << offbit
                d0 = lastc[ord(nt0)]
 
                n1, hd1, nt1 = path_rdbg[idx+1]
                h1 = lastc[ord(hd1)] << offbit
                d1 = lastc[ord(nt1)]
 
                k12 = np.asarray([n0, h0|d0, n1, h1|d1])

                # remove the repeat in the same sequences
                if k12 not in visit:
                    visit.add(k12)
                else:
                    continue

                try:
                    rdbg[k12] += 1
                except:
                    rdbg[k12] = 1

            n = len(seq)
        N += n
        if N > Ns:
            break

    print('rdbg size 1665', rdbg.size)
    _oname = qry + '_rdbg_weight.xyz'
    _o = open(_oname, 'w')
    for k12 in rdbg.iteritems():
        #n0, n1 = k12
        (n0, hd0, n1, hd1), val = k12
        xyz = '%d_%d\t%d_%d\t%d\n'%(n0, hd0, n1, hd1, val)
        _o.write(xyz)

    _o.close()

    # call the mcl for clustering
    os.system('mcl %s --abc -I 1.5 -te 8 -o %s.mcl'%(_oname, _oname))

    del rdbg
    del kmer_dict
    gc.collect()

    spec['vtype'] = nb.uint32
    spec['values'] = spec['vtype'][:]
    if jit and spec:
        hashfunc_jit = nb.jitclass(spec)(hashfunc)
    else:
        hashfunc_jit = hashfunc
 
    label_dct = hashfunc_jit(capacity=2**20, ksize=2, ktype=spec['ktype'], vtype=spec['vtype'])

    flag = 0
    f = open(_oname+'.mcl', 'r')
    for i in f:
        j = i[:-1].split('\t')
        for k in j:
            ky = tuple(map(int, k.split('_')[:2]))
            label_dct[ky] = flag

        flag += 1

    f.close()

    # add the rest kmer
    f = open(_oname, 'r')
    for i in f:
        j, k = i[:-1].split('\t')[:2]

        kj = tuple(map(int, j.split('_')[:2]))
        if not label_dct.has_key(kj):
            #label_dct[j] = flag
            label_dct[kj] = flag
            flag += 1

        kk = tuple(map(int, k.split('_')[:2]))
        if not label_dct.has_key(kk):
            label_dct[kk] = flag
            flag += 1


    N = 0
    print('label_dct', label_dct.size)
    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)

        for seq in [seq_fw]:
            starts = [0]
            labels = [-1]
            for idx, k, hd, nt in seq2ns_(seq, kmer, bits):

                h = lastc[ord(hd)] << offbit
                d = lastc[ord(nt)]
                kk = (k, h|d)
 
                if label_dct.has_key(kk):
                    label = label_dct[kk]
                    if starts[-1] < idx:

                        if labels[-1] != label:
                            labels.append(label)
                            starts.append(idx+kmer)
                        # the same region, just extend it.
                        else:
                            starts[-1] = idx + kmer

            for idx in xrange(1, len(starts)):
                print('%s\t%d\t%d\t%s\t%d'%(i.id, starts[idx-1], starts[idx], '+', labels[idx]))

        N += len(seq_fw)
        if N > Ns:
            break

    print('end', 1798)
    #return kmer_dict
    return label_dct

# recover the sequence from the compressed path
def recover(path, dbg):
    pass


# print the manual
def manual_print():
    print('Usage:')
    print('  pyhton this.py -i qry.fsa -k 10 -n 1000000')
    print('Parameters:')
    print('  -i: query sequences in fasta format')
    print('  -k: kmer length')
    print('  -n: length of query sequences for dbg')

def entry_point(argv):

    args = {'-i': '', '-k': '50', '-n': '1000000', '-r': '', '-d': ''}
    N = len(argv)

    for i in xrange(1, N):
        k = argv[i]
        if k in args:
            v = argv[i + 1]
            args[k] = v
        elif k[:2] in args and len(k) > 2:
            args[k[:2]] = k[2:]
        else:
            continue

    # bkt, the breakpoint
    qry, kmer, Ns, bkt, dbs = args['-i'], int(args['-k']), int(eval(args['-n'])), args['-r'], args['-d']
    if not qry:
        seq = 'ACCCATCGGGCTAAACCCCCCCCCCGATCGATCGAC'
        #seq = 'AAAAAAAAAAGAAAAAAAAAATAAAAAAAAAACAAAAAAAAAA'
        seq = 'AAAACCCCAATACCCCATAACCCC'
        kmer = 4
        a0 = [(k2n_(seq[elem:elem+kmer]), seq[elem-1:elem], seq[elem+kmer:elem+kmer+1]) for elem in xrange(len(seq)-kmer+1)]
        a1 = [elem[:] for elem in seq2ns_(seq, kmer)]
        print(a0 == a1) 
        print(a0[-105:])
        print(a1[-105:])

        # test 
        try:
            N = int(eval(sys.argv[1]))
        except:
            N = 10**7

        mkey = 5
        print('test', N)

        # the spec for jitclass
        spec = {}
        spec['capacity'] = nb.int64
        spec['load'] = nb.float32
        spec['size'] = nb.int64
        spec['ksize'] = nb.int64
        spec['ktype'] = nb.uint64
        spec['keys'] = spec['ktype'][:]
        spec['vtype'] = nb.uint16
        spec['values'] = spec['vtype'][:]
        spec['counts'] = nb.uint8[:]

        pypy = platform.python_implementation().lower() 
        if pypy == 'pypy':
            oakht_jit = oakht
        else:
            oakht_jit = nb.jitclass(spec)(oakht)

        clf = oakht_jit(capacity=int(N * 1.34), ksize=1, ktype=spec['ktype'], vtype=spec['vtype'])
        string = 'ATGC' * (N // 4)
        seq = np.frombuffer(string.encode(), dtype=np.uint8)
        np.random.shuffle(seq)

        res = kmer2dict(seq, clf, 27)
        print('finish', clf.size, (clf.counts>0).sum())
        raise SystemExit()


        clf = oakht_jit(capacity=N, ksize=mkey, ktype=spec['ktype'], vtype=spec['vtype'])
        #clf = oakht_jit(capacity=int(N * 1.75), ksize=mkey, ktype=spec['ktype'], vtype=spec['vtype'])

        print('initial finish', 'python version', pypy)
        def oa_test(N, k, clf):

            x = [randint(0, N) for elem in range(N)]
            for i in range(N-k+1):
                clf.push(x[i:i+k], i)
                #clf.push(np.random.randint(0, N, k), i)

            flag = 0
            for i in range(N-k+1):
                #clf.push(x[i:i+k], i)
                #clf.push(np.random.randint(0, N, k), i)
                if not clf.has_key(x[i:i+k]):
                    flag += 1

            print('x err', flag)

            # check random generated array
            flag = 0
            y = [randint(0, N) for elem in range(N)]
            for i in range(N-k+1):
                if clf.has_key(y[i:i+k]):
                    flag += 1

            print('y err', flag)

            flag = 0
            for kv in clf.iteritems():
                print('item is', kv)
                if flag > 5:
                    break
                flag += 1

        @nb.njit
        def oa_test_jit(N, k, clf):
            x = np.random.randint(0, N, N)
            for i in range(N-k+1):
                clf.push(x[i:i+k], i)
                #clf.push(np.random.randint(0, N, k), i)

            flag = 0
            for i in range(N-k+1):
                #clf.push(x[i:i+k], i)
                #clf.push(np.random.randint(0, N, k), i)
                if not clf.has_key(x[i:i+k]):
                    flag += 1

            print('x err', flag)

            # check random generated array
            flag = 0
            y = np.random.randint(0, N, N)
            for i in range(N-k+1):
                if clf.has_key(y[i:i+k]):
                    flag += 1

            print('y err', flag)

            flag = 0
            for kv in clf.iteritems():
                print('item is', kv)
                if flag > 5:
                    break
                flag += 1

        print('numba version')
        if pypy == 'pypy':
            oa_test(N, mkey, clf)
        else:
            oa_test_jit(N, mkey, clf)
        raise SystemExit()

    if dbs:
        #print('recover from', bkt)
        # convert sequence to path and build the graph
        dct = seq2graph(qry, kmer=kmer, bits=5, Ns=Ns, saved=dbs, hashfunc=oakht)
    else:
        # build the rdbg, convert sequence to path, and build the graph
        #dct = seq2dbg(qry, kmer, 5, Ns, rec=bkt)
        kmer_dict = seq2rdbg(qry, kmer, 5, Ns, rec=bkt)
        #for i in kmer_dict:
        #    print('kmer dict size', qry, len(kmer_dict), kmer, n2k_(i, kmer, 5))
        #    if bkt:
        #        print('bkt is', bkt)
        dct = seq2graph(qry, kmer=kmer, bits=5, Ns=Ns, kmer_dict=kmer_dict, hashfunc=oakht)

    return 0
    #return dct


def target(*args):
    return entry_point, None

if __name__ == "__main__":

    entry_point(sys.argv)
