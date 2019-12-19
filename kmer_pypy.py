#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# CreateTime: 2016-05-09 15:46:28

import sys
from Bio import SeqIO
from array import array
from _numpypy import multiarray as np

# the last char of the kmer
# A: 0001
# T: 0010
# G: 0100
# C: 1000
# N: 0000
lastc = np.zeros(256, dtype='int8')
lastc[ord('a')] = lastc[ord('A')] = 0b0001
lastc[ord('t')] = lastc[ord('T')] = 0b0010
lastc[ord('g')] = lastc[ord('G')] = 0b0100
lastc[ord('c')] = lastc[ord('C')] = 0b1000


# count 1 of the binary number
def nbit(n):
    x = n - ((n >>1) &033333333333) - ((n >>2) &011111111111)
    return ((x + (x >>3)) &030707070707) %63


# convert dna kmer to number
# a:00, t:11, g:01, c:10
#alpha = array('i', [0] * 256)
alpha = np.zeros(256, dtype='int8')
alpha[ord('a')] = alpha[ord('A')] = 0b00
alpha[ord('t')] = alpha[ord('T')] = 0b11
alpha[ord('g')] = alpha[ord('G')] = 0b01
alpha[ord('c')] = alpha[ord('C')] = 0b10


def k2n_(kmer):
    n, N = len(kmer), 0
    for i in xrange(n):
        c = alpha[ord(kmer[i])]
        N += (c << (i*2))
    return N
    
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


def seq2ns_(seq, k=12):
    n = len(seq)
    if n < k:
        #return -1
        yield -1, 0

    Nu = k2n_(seq[:k])
    yield Nu, 0
    shift = k*2-2
    for i in xrange(k, n):
        #c = alpha[ord(seq[i])]
        j = ord(seq[i])
        c = alpha[j]
        Nu = ((Nu >> 2) | (c << shift))
        yield Nu, lastc[j]
# print the manual
def manual_print():
    print 'Usage:'
    print '  fsearch -p blastp -i qry.fsa -d db.fsa'
    print 'Parameters:'
    print '  -p: program'
    print '  -i: query sequences in fasta format'
    print '  -l: start index of query sequences'
    print '  -u: end index of query sequences'
    print '  -L: start index of reference'
    print '  -U: end index of reference'
    print '  -d: ref database'
    print '  -D: index of ref, if this parameter is specified, only this part of formatted ref will be searched against'
    print '  -o: output file'
    print '  -O: write mode of output file. w: overwrite, a: append'
    print '  -s: spaced seed in format: 1111,1110,1001.. etc'
    print '  -r: reduced amino acid alphabet in format: AST,CFILMVY,DN,EQ,G,H,KR,P,W'
    print '  -v: number of hits to show'
    print '  -e: expect value'
    print '  -m: max ratio of pseudo hits that will trigger stop'
    print '  -j: distance between start sites of two neighbor seeds, greater will reduce the size of database'
    print '  -t: filter high frequency kmers whose counts > t'
    print '  -F: Filter query sequence'
    print '  -M: bucket size of hash table, reduce this parameter will reduce memory usage but decrease sensitivity'
    print '  -c: chunck size of reference. default is 50K which mean 50K sequences from reference will be used as database'
    print '  -T: tmpdir to store tmp file. default ./tmpdir'


def entry_point(argv):

    seeds = '111111'
    aa_nr = 'AST,CFILMVY,DN,EQ,G,H,KR,P,W'
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
    kmer = min(max(1, kmer), 31)
    size = int(pow(4, kmer)+1)
    print('size', size)
    #kmer_dict = array('l', [0]) * size
    kmer_dict = np.zeros(size, dtype='int8')
   
    #for i in xrange(size):
    #    kmer_dict[i] = i

    N = 0

    for i in SeqIO.parse(qry, 'fasta'):
        seq = i.seq
        n = len(seq)
        for k, d in seq2ns_(seq, kmer):
            kmer_dict[k] |= d
               
        N += n
        if N > Ns:
            break
    print('dct size', len(kmer_dict), 'seq', N, 'min seq', Ns)
    return 0


def target(*args):
    return entry_point, None

if __name__ == "__main__":

    entry_point(sys.argv)
