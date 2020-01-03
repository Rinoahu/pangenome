#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# CreateTime: 2016-05-09 15:46:28

import sys
from Bio import SeqIO
from array import array
from bisect import bisect_left
import math
import mmap
import networkx as nx
import gc
try:
    from _numpypy import multiarray as np
except:
    import numpy as np

try:
    xrange = xrange
except:
    xrange = range


# blist in pure python
rng = lambda x: (((x * 279470273) % 4294967291) * 279470273) % 4294967291
# this is blist data structure
class Node:
    def __init__(self, size = 0, left = [], right = [], next = None):
        self.size = size
        self.left = left
        self.right = right
        self.next = next

class leaf:
    def __init__(self, vals = []):
        self.vals = vals
        self.size = len(self.vals)

class Blist:

    def __init__(self, chunk = 2 ** 10):
        self.root = Node(0, [], [], None)
        self.full = chunk
        self.double = self.full * 2
        self.half = self.full // 2
        self.quarter = self.full // 4

    # get the size or rank of child
    def l_size(self, node):
        return isinstance(node.left, Node) and node.left.size or len(node.left)

    def r_size(self, node):
        return isinstance(node.right, Node) and node.right.size or len(node.right)

    def calc_size(self, node):
        return self.l_size(node) + self.r_size(node)

    # give the key and locate the node
    def locate(self, x):
        node = self.root
        idx = x
        visit = []
        while isinstance(node, Node):
            l_size = self.l_size(node)
            if idx < l_size:
#           if idx < l_size or idx == l_size == 0:
                pred_node = node
                pred = 'left'
#               node = node.left
                node = pred_node.left
            else:
                idx -= l_size
                pred_node = node
                pred = 'right'
                node = pred_node.right

            visit.append((pred_node, pred))

#       print 'depth is', len(visit)
        return visit, idx

    # rotation
    #      a     b
    #     / \       / \
    #    b   c <=> d   a
    #   / \       / \
    #  d   e     e   c
    def rotate_right(self, a):
        b = a.left
        if isinstance(b, Node):
            e = b.right
            a.left = e
            b.right = a
            a.size = self.calc_size(a)
            b.size = self.calc_size(b)

            return b
        else:
            return a

    def rotate_left(self, b):
        a = b.right
        if isinstance(a, Node):
            e = a.left
            b.right = e
            a.left = b
            b.size = self.calc_size(b)
            a.size = self.calc_size(a)

            return a
        else:
            return b

    # split the node
    #    a        a
    #   / \  ->  / \
    # [x] [y]  [x]  c
    #          / \
    #     [:y/2] [y/2:]
    def split(self, y):
        N = len(y)
        if N > self.full:
            M = N // 2
            c = Node(N, y[: M], y[M: ], None)
            return c
        else:
            return y

    # split the node
    #    a         a
    #   / \       / \
    # [z]  c -> [z] [x,y]
    #     / \
    #   [x] [y]
    def merge(self, c):
        a, b = self.l_size(c), self.r_size(c)
#       a, b = c.left, c.right
        if (a < self.quarter and b < self.quarter) or a + b < self.half:
#       if len(a) < self.quarter and len(b) < self.quarter or len(a) + len(b) < self.half:
#           a.extend(b)
#           return a
#           c.left.extend(c.right)
#           return c.left
            return c.left + c.right
        else:
            return c

    # make the node balance
    def blance(self, x):
        if isinstance(x, list):
            return x
        else:
            # treap like algorithm
#           a  = rng(self.l_size(x))
#           if a < 1431655763:
#               return self.rotate_left(x)
#           elif a > 2863311526:
#               return self.rotate_right(x)
#           else:
#               return x

            # the b-btree algorithm
            d = self.l_size(x) - self.r_size(x)
            if abs(d) < self.double:
                return x
            elif d < 0:
                return self.rotate_left(x)
            else:
                return self.rotate_right(x)

    def insert(self, x, y):
        visit, idx = self.locate(x)
        pre_node, pre_pred = visit.pop()
        if pre_pred is 'left':
            vals = pre_node.left
        else:
            vals = pre_node.right

        # insert the value to list
        vals.insert(idx, y)
        # split the list if necessary
        split_node = self.split(vals)

        if pre_pred is 'left':
            pre_node.left = split_node
        else:
            pre_node.right = split_node

#       pre_node.size = self.calc_size(pre_node)
        pre_node.size += 1
        # do the rotate if necessary
        while visit:
            new_node, new_pred = visit.pop()
            if new_pred is 'left':
                new_node.left = self.blance(pre_node)
            else:
                new_node.right = self.blance(pre_node)
#           new_node.size = self.calc_size(new_node)
            new_node.size += 1
            pre_node  = new_node

    # common features in blist, most are the same as in list
    def __len__(self):
        return self.root.size

    def size(self):
        visit = [self.root]
        flag = 0
        while visit:
#           print 'adding'
            node = visit.pop()
            if isinstance(node, Node):
#               print 'test', flag
                flag += 1
            if isinstance(node.left, Node):
                visit.append(node.left)
            if isinstance(node.right, Node):
                visit.append(node.right)
        return flag

    def __setitem__(self, x, y):
        if x < 0:
#           x += self.root.size
            x += self.__len__()
        visit, idx = self.locate(x)
        pre_node, pre_pred = visit.pop()
        if pre_pred is 'left':
            pre_node.left[idx] = y
        else:
            pre_node.right[idx] = y

    def __getitem__(self, x):
        if x < 0:
#           x += self.root.size
            x += self.__len__()
        visit, idx = self.locate(x)
        pre_node, pre_pred = visit.pop()
        if pre_pred is 'left':
            return pre_node.left[idx]
        else:
            return pre_node.right[idx]

    def __getslice__(self, x, y):
        if x < 0:
#           x += self.root.size
            x += self.__len__()
        if y < 0:
#           y += self.root.size
            y += self.__len__()
#       return [self[elem] for elem in xrange(x, y) if elem < self.__len__()]
        return [self[elem] for elem in xrange(x, min(y, self.__len__()))]

    # the slice method, same as list
    def __setslice__(self, x, y, z):
        for i, j in zip(xrange(x, y), z):
            self[i] = j
    # the append method, add elem to the last
    def append(self, x):
        self.insert(self.__len__(), x)

    # the extend methd, add elems to the last
    def extend(self, x):
        for i in x:
            self.append(i)

    def __delitem__(self, x):
        if x < 0:
            x += self.root.size

        visit, idx = self.locate(x)
        pre_node, pre_pred = visit.pop()
        if pre_pred is 'left':
            vals = pre_node.left
        else:
            vals = pre_node.right

        # del the value from list
        del vals[idx]
#       pre_node.size = self.calc_size(pre_node)
        pre_node.size -= 1
        if visit:
            pre_node = self.merge(pre_node)
        # do the rotate if necessary
        while visit:
            new_node, new_pred = visit.pop()
            if new_pred is 'left':
                new_node.left = self.blance(pre_node)
            else:
                new_node.right = self.blance(pre_node)
#           new_node.size = self.calc_size(new_node)
            new_node.size -= 1
            pre_node  = new_node

    def __str__(self):
        print str(self[:])

    def __repr__(self):
        return str(self[:])


# memmap function for pypy
def memmap(fn, mode='w+', shape=None, dtype='int8'):
    if dtype == 'int8' or dtype == 'uint8':
        stride = 1
    elif dtype == 'float16' or dtype == 'int16' or dtype == 'uint16':
        stride = 2
    elif dtype == 'float32' or dtype == 'int32':
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
    return np.frombuffer(buf, dtype=dtype)


# open addressing hash table for kmer count
class ht:
    def __init__(self, capacity=1023, load_factor = .6666667, key_type='uint64', val_type='uint16'):

        self.capacity = capacity
        # load factor is 2/3
        self.load = load_factor
        self.size = 0
        self.null = 2**64-1
        # for big, my own mmap based array can be used
        self.keys = np.empty(capacity, dtype=key_type)
        self.values = np.empty(capacity, dtype=val_type)
        self.keys[:] = self.null

    def resize(self):
        N = self.capacity
        M = N * 2
        null = self.null
        keys = np.empty(M, dtype='uint64')
        values = np.empty(M, dtype='uint16')
        keys[:] = null
        self.capacity = M

        # re-hash
        for i in xrange(N):
            key = self.keys[i]
            if key != null:
                value = self.values[i]
                # new hash
                j, k = key % M, 1
                while keys[j] != null:
                    j = (j + k*k) % M
                    k += 1

                keys[j] = key
                values[j] = value
            else:
                continue

        self.keys = keys
        self.values = values
        gc.collect()

    def pointer(self, key):
        M = self.capacity
        null = self.null
        j, k = key % M, 1
        while null != self.keys[j] != key:
            j = (j + k*k) % M

        return j


    def __setitem__(self, key, value):
        j = self.pointer(key)
        self.keys[j] = key
        self.values[j] = value
        self.size += 1
        # if too many elements
        if self.size * 1. / self.capacity > self.load:
            self.resize()
            print('resize')

    def __getitem__(self, key):
        j = self.pointer(key)
        #print('key', key, 'target', j, self.keys[j])
        if key == self.keys[j]:
            return self.values[j]
        else:
            raise KeyError

    def __delitem__(self, key):
        j = self.pointer(key)
        if key == self.keys[j]:
            self.keys[j] = self.null
            self.size -= 1
        else:
            raise KeyError

    def has_key(self, key):
        j = self.pointer(key)
        return key == self.keys[j] and True or False

    def __iter__(self):
        null = self.null
        for i in self.keys:
            if i != null:
                yield i

# count 1 of the binary number
#def nbit(n):
#    x = n - ((n >> 1) & 033333333333) - ((n >> 2) & 011111111111)
#    return ((x + (x >> 3)) & 030707070707) % 63

def nbit(n):
    x = n - ((n >> 1) & 3681400539) - ((n >> 2) & 1227133513)
    return ((x + (x >> 3)) & 3340530119) % 63


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
print('offbit', offbit, 'low bin', bin(lowbit))

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
    n, s = N, []
    for i in xrange(K):
        c = beta[n % bit]
        n //= bit
        s.append(c)
    return ''.join(s)

    
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


def seq2ns_(seq, k=12, bit=5):
    n = len(seq)
    if n < k:
        #yield -1, '0', '0'
        yield -1, '#', '$'

    Nu = k2n_(seq[:k])
    #yield Nu, '0', seq[k]
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

# bisect based query
# xs is the sorted array
# x is the query
def query(xs, x):
    idx = bisect_left(xs, x)
    return x in xs[idx:idx+1]

# function to compress genome sequence
def seq2dbg(qry, kmer=13, bits=5, Ns=1e6):
    kmer = min(max(1, kmer), 31)
    size = int(pow(bits, kmer)+1)
    kmer_dict = memmap('tmp.npy', shape=size, dtype='int16')
   
    #for i in xrange(size):
    #    kmer_dict[i] = i

    N = 0
    for i in SeqIO.parse(qry, 'fasta'):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)
        #print('seq', seq_fw[:10], seq_rv[:10])
        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            for k, hd, nt in seq2ns_(seq, kmer, bits):
                #km = n2k_(k, kmer)
                #if km =='CCCC':
                #    print(km, hd, nt)

                h = lastc[ord(hd)] << offbit
                d = lastc[ord(nt)]
                kmer_dict[k] |= (h | d) 
 
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
    for i in xrange(len(kmer_dict)):
        hn = kmer_dict[i]
        #if kmer_dict[i] > 0:
        if hn > 0:
            km = n2k_(i, kmer)
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
        path = []
        #print('seq', seq_fw)
        #for seq in [seq_fw, seq_rv]:
        for seq in [seq_fw]:
            skip = p0 = p1 = 0
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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
        print('path', len(path), 'seq', n)
        print(path[:])
        n = len(seq)
        N += n
        if N > Ns:
            break
    print('Graph size', Graph.size(), 'edge', len(Graph.edges()), 'node', len(Graph.nodes()))
    #for n0, n1 in Graph.edges():
    #    print('edge\t%d\t%d'%(n0, n1))

# print the manual
def manual_print():
    print('Usage:')
    print('  pyhton this.py -i qry.fsa -k 10 -n 1000000')
    print('Parameters:')
    print('  -i: query sequences in fasta format')
    print('  -k: kmer length')
    print('  -n: length of query sequences for dbg')

def entry_point(argv):

    args = {'-i': '', '-k': '50', '-n': '1000000'}
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

    qry, kmer, Ns = args['-i'], int(args['-k']), int(eval(args['-n']))
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
        raise SystemExit()

    seq2dbg(qry, kmer, 5, Ns)
    return 0


def target(*args):
    return entry_point, None

if __name__ == "__main__":

    entry_point(sys.argv)
