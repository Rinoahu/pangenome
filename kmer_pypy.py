#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# CreateTime: 2016-05-09 15:46:28

import sys
from Bio import SeqIO
from array import array
from _numpypy import multiarray as np
import mmap


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


# the last char of the kmer
# A: 0001
# T: 0010
# G: 0100
# C: 1000
# N: 0000
lastc = np.zeros(256, dtype='int8')
lastc[ord('a')] = lastc[ord('A')] = 0b1
lastc[ord('t')] = lastc[ord('T')] = 0b10
lastc[ord('g')] = lastc[ord('G')] = 0b100
lastc[ord('c')] = lastc[ord('C')] = 0b1000
lastc[ord('n')] = lastc[ord('N')] = 0b10000

lastc_r = ['0']* 0b10001
lastc_r[0b1] = 'A'
lastc_r[0b10] = 'T'
lastc_r[0b100] = 'G'
lastc_r[0b1000] = 'C'
lastc_r[0b10000] = 'N'
lastc_r = ''.join(lastc_r)


# count 1 of the binary number
def nbit(n):
    x = n - ((n >>1) &033333333333) - ((n >>2) &011111111111)
    return ((x + (x >>3)) &030707070707) %63


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
        #return -1
        yield -1, '0', '0'

    Nu = k2n_(seq[:k])
    yield Nu, '0', seq[k]

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
    yield Nu, hd, '0'

# print the manual
def manual_print():
    print 'Usage:'
    print '  pyhton this.py -i qry.fsa -k 10 -n 1000000'
    print 'Parameters:'
    print '  -i: query sequences in fasta format'
    print '  -k: kmer length'
    print '  -n: length of query sequences for dbg'

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
        a0 = [(k2n_(seq[elem:elem+12]), seq[elem-1:elem], seq[elem+12:elem+13]) for elem in xrange(len(seq)-12+1)]
        a1 = [elem[:] for elem in seq2ns_(seq)]
        print(a0 == a1) 
        print(a0[-5:])
        print(a1[-5:])
        raise SystemExit()

    kmer = min(max(1, kmer), 31)
    bits = 5
    size = int(pow(bits, kmer)+1)
    print('size', size)
    #kmer_dict = array('l', [0]) * size
    #kmer_dict = np.zeros(size, dtype='int16')
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
                h = lastc[ord(hd)] << 5
                d = lastc[ord(nt)]
                kmer_dict[k] |= (h | d) 
 
            N += n
        print('N is', N)
        if N > Ns:
            break

    out_deg = 0
    for i in xrange(len(kmer_dict)):
        if kmer_dict[i] > 0:
            km = n2k_(i, kmer)
            hn = kmer_dict[i]
            pr = nbit(hn >> 5)
            sf = nbit(hn & 0b11111)
            print('%s\t%s\t%s'%(km, pr, sf))
            out_deg += (sf > 1)

    print('dct size', len(kmer_dict), 'seq', N, 'min seq', Ns, 'branch', out_deg, 'rate', out_deg*100./N)
    return 0


def target(*args):
    return entry_point, None

if __name__ == "__main__":

    entry_point(sys.argv)
