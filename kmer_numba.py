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

            #self.njit = lambda x: lambda y: y
            def njit(inline='never', spec={}):
                return lambda x: x

            self.njit = njit
            self.jitclass = njit

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


# convert sequences to bit array
def seq2bytes(fn):
    seq_bytes = np.asarray(np.memmap(fn, mode='r+', dtype='uint8'))
    return seq_bytes

# readline
@nb.njit
#def readline_jit_(seq_bytes):
def readline_jit_(seq_bytes, offset=0):
    start = end = 0
    #for end in range(len(seq_bytes)):
    for end in range(offset, len(seq_bytes)):
        if seq_bytes[end] == 10:
            yield start, end + 1
            start = end + 1
    if end > start > 0:
        yield start, end + 1

# sequence parse
@nb.njit
#def seqio_jit_(seq_bytes, isfasta=True):
def seqio_jit_(seq_bytes, offset=0, isfasta=True):

    if isfasta:

        # the pointer of the sequence
        ptr = np.empty(1, dtype=nb.uint64)

        qid = np.empty(1, dtype=nb.uint8)
        qid[0] = 0
        seq = np.empty(2**27, dtype=nb.uint8)
        empty = np.empty(2**27, dtype=nb.uint8)

        # pointer for sequence
        start = end = 0
        #for st, ed in readline_jit_(seq_bytes):
        for st, ed in readline_jit_(seq_bytes, offset=offset):
            ptr[0] = st
            line = seq_bytes[st: ed]
            if line[0] == 62:
                if qid[0] == 62:
                    #yield qid, seq[: end]
                    yield qid, seq[: end], ptr

                qid = line[:-1]
                start = end = 0
            else:
                end += (len(line) - 1)
                if end > len(seq):
                    seq = np.append(seq, empty)
                
                seq[start: end] = line[:-1]
                start = end

        if qid[0] == 62:
            #yield qid, seq[: end]
            yield qid, seq[: end], ptr

    else:
        flag = 0
        for st, ed in readline_jit_(seq_bytes):
            ptr[0] = ed
            line = seq_bytes[st: ed]
            if line[0] == 64:
                head = line[:-1]
                flag = 1
            elif flag == 1:
                seq = line[:-1]
                #yield head, seq
                yield head, seq, ptr
                flag = 0
            else:
                continue

# reverse the sequence
tab_rev_bytes = np.empty(256, dtype='uint8')
tab_rev_bytes[:] = ord('N')
for i, j in zip('ATGCN', 'TACGN'):
    tab_rev_bytes[ord(i)] = ord(j)
    tab_rev_bytes[ord(i.lower())] = ord(j)

@nb.njit
def reverse_jit_(seq_bytes, tab_rev=tab_rev_bytes):
    N = len(seq_bytes)
    seq_rv = np.empty(N, dtype=nb.uint8)
    for i in xrange(N):
        seq_rv[i] = tab_rev[seq_bytes[N-1-i]]

    return seq_rv


# test
@nb.njit
def parse_test(seq_types):
    chk = flag = 0
    for qid, seq, ptr in seqio_jit_(seq_types):

        if flag % 10**6 == 0:
            print('iter', flag, 'current seq', len(seq), 'total seq', chk/1e9)

        chk += len(seq)
        flag += 1

    return 0

# save the dict to disk
def dump(clf, fn='./tmp', offset=0):
    #print('213 clf size', clf.size, fn)
    fn = fn.endswith('.npz') and fn[:-4] or fn

    capacity = clf.capacity
    load_factor = clf.load
    size = clf.size
    ksize = clf.ksize
    parameters = np.asarray([capacity, load_factor, size, ksize, offset], dtype='uint64')
    np.savez_compressed(fn, parameters=parameters, keys=clf.keys, values=clf.values, counts=clf.counts)
    #np.savez(fn, parameters=parameters, keys=clf.keys, values=clf.values, counts=clf.counts)

    return 0

# load the dict from disk
def load_on_disk(fn='./tmp', mmap='r+'):

    dtypes = {
            np.empty(1, 'uint64').dtype: nb.uint64,
            np.empty(1, 'uint32').dtype: nb.uint32,
            np.empty(1, 'uint16').dtype: nb.uint16,
            np.empty(1, 'uint8').dtype: nb.uint8,
            np.empty(1, 'int64').dtype: nb.int64,
            np.empty(1, 'int32').dtype: nb.int32,
            np.empty(1, 'int16').dtype: nb.int16,
            np.empty(1, 'int8').dtype: nb.int8
            }

    loaded = np.load(fn)

    parameters = loaded['parameters']
    capacity, load_factor, size, ksize, offset = parameters

    keys = loaded['keys']
    values = loaded['values']
    counts = loaded['counts']

    ktype = dtypes[keys.dtype]
    vtype = dtypes[values.dtype]

    clf = init_dict(hashfunc=oakht, capacity=1, ksize=ksize, ktype=ktype, vtype=vtype, jit=True)
    #clf = init_dict(hashfunc=oakht, capacity=1, ksize=ksize)
    clf.capacity = capacity
    clf.load = load_factor
    clf.size = size

    clf.keys = keys
    clf.values = values
    clf.counts = counts

    return offset, clf


# jit version
class oakht0:
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
 
    # clean data
    def destroy(self):
        #self.counts[:] = 0
        keys_old, values_old, counts_old = self.keys, self.values, self.counts

        keys = np.empty(self.ksize, dtype=keys_old.dtype)
        values = np.empty(1, dtype=values_old.dtype)
        counts = np.zeros(1, dtype=counts_old.dtype)
        self.keys, self.values, self.counts = keys, values, counts

        del keys_old, values_old, counts_old
        #gc.collect()

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
                #print('k size', ks)
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

# add multiple values
class oakht:
    def __init__(self, capacity=1024, load_factor = .75, ksize=1, ktype=nb.int64, vsize=1, vtype=nb.int64):

        self.capacity = self.find_prime(capacity)
        self.load = load_factor
        self.size = 0
        self.ksize = ksize
        self.vsize = vsize
        N = self.capacity

        self.keys = np.empty(N * ksize, dtype=ktype)
        self.values = np.empty(N * vsize, dtype=vtype)
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
 
    # clean data
    def destroy(self):
        #self.counts[:] = 0
        keys_old, values_old, counts_old = self.keys, self.values, self.counts

        keys = np.empty(self.ksize, dtype=keys_old.dtype)
        values = np.empty(1, dtype=values_old.dtype)
        counts = np.zeros(1, dtype=counts_old.dtype)
        self.keys, self.values, self.counts = keys, values, counts

        del keys_old, values_old, counts_old
        #gc.collect()

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
        vs = self.vsize
        keys_old, values_old, counts_old = self.keys, self.values, self.counts

        # get new arrays
        self.capacity = self.find_prime(nb.longlong(N * 1.62))
        M = self.capacity

        keys = np.empty(M * ks, dtype=keys_old.dtype)
        values = np.empty(M, dtype=values_old.dtype)
        counts = np.zeros(M, dtype=counts_old.dtype)

        for i in xrange(N):
            if counts_old[i] > 0:
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

                iv, jv = i * vs, j * vs
                values[jv: jv+vs] = values_old[iv: iv+vs]

                counts[j] = counts_old[i]

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

    def push(self, key, value, kstart=0, vstart=0):
        j = self.pointer(key, kstart)
        ks = self.ksize
        jk = j * ks
        if self.counts[j] == 0:
            self.size += 1
            self.keys[jk: jk+ks] = key[kstart: kstart+ks]

        vs = self.vsize
        jv = j * vs
        self.values[jv: jv+vs] = value[vstart: vstart+vs]
        self.counts[j] = min(self.counts[j] + 1, 255)

        # if too many elements
        if self.size * 1. / self.capacity > self.load:
            #a, b =  self.size, self.capacity
            self.resize()
            #print('before resize', a, b, 'after resize', self.size, self.capacity)

    def __setitem__(self, key, value, kstart=0, vstart=0):
        self.push(key, value, kstart, vstart)

    def get(self, key, start=0):
        j = self.pointer(key, start)
        ks = self.ksize
        vs = self.vsize
        jk = j * ks
        jv = j * vs
        if self.eq(self.keys, jk, key, start, ks):
            return self.values[jv: jv+vs]
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
        vs = self.vsize
        for i in xrange(0, self.counts.shape[0]):
            if self.counts[i] > 0:
                iv = i * vs
                yield self.values[iv: iv+vs]


    def iteritems(self):
        ks = self.ksize
        vs = self.vsize
        for i in xrange(0, self.counts.shape[0]):
            ik = i * ks
            iv = i * vs
            if self.counts[i] > 0:
                #print('k size', ks)
                yield self.keys[ik:ik+ks], self.values[iv: iv+vs]

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
    return s


# convert a sequence to numeric array, very slow
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


# convert a sequence to numeric array, optimized
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


# check how many indegree and outdegree a node has
def query(ht, i):
    try:
        hn = ht.get([i])
    except:
        return False

    if hn > 0:
        pr = nbit(hn >> offbit)
        sf = nbit(hn & lowbit)
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


# load dbg on disk
def load_dbg(saved, kmer_dict):

    dump_keys, dump_fk = memmap(saved + '_dump_key.npy', 'a+', dtype='uint64')
    dump_values, dump_fv = memmap(saved + '_dump_val.npy', 'a+', dtype='uint16')
    dump_counts, dump_fc = memmap(saved + '_dump_cnt.npy', 'a+', dtype='uint8')

    for i in xrange(dump_values.shape[0]):
        if query(dump_values, i):
            key, val = dump_keys[i], dump_values[i]
            kmer_dict[key] = val

    dump_fk.close()
    dump_fv.close()
    dump_fc.close()

    return 0


#@nb.njit(inline='always')
def k2n_jit(kmer, bit=5, alpha=alpha):
    #N = nb.ulonglong(0)
    N = idx = 0
    #for i in xrange(len(kmer)):
    #idx = 0
    for i in kmer:
        c = nb.ulonglong(alpha[i])
        N += c * bit ** idx
        idx += 1

    return N

#@nb.njit(inline='always')
k2n_jit_ = nb.njit(inline='always')(k2n_jit)


@nb.njit
def seq2ns_jit_(seq, k=12, bit=5, alpha=alpha):
    # '#' is 35, '$' is 36
    output = np.empty(4, dtype=np.uint64)
    n = len(seq)
    if n > k:
        Nu = k2n_jit_(seq[:k])
        idx = 0
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

# add kmer to the dbg
@nb.njit(inline='always')
def add_kmer(kmer_dict, key, empty, idx, Nu, hd, nc, lastc=lastc, offbit=offbit):
    key[0], hd, nt = Nu, hd, nc
    h = lastc[hd] << offbit
    d = lastc[nt]
    if kmer_dict.has_key(key):
        val = kmer_dict.get(key)
        kmer_dict.push(key, val | h | d)
    else:
        empty[0] = h|d
        #kmer_dict.push(key, np.asarray([h|d], dtype=nb.uint16))
        kmer_dict.push(key, empty)


# build dbg from kmer
#def kmer2dict(seq, kmer_dict, kmer=12, bit=5, offbit=offbit, lastc=lastc, alpha=alpha):
def build_dbg(seq, kmer_dict, kmer=12, bit=5, offbit=offbit, lastc=lastc, alpha=alpha):

    k = kmer
    key = np.empty(1, dtype=np.uint64)
    empty = np.empty(1, dtype=np.uint16)

    # '#' is 35, '$' is 36
    #output = np.empty(4, dtype=np.uint64)
    n = len(seq)
    if n > k:
        idx = 0
        Nu = k2n_jit_(seq[:k])
        hd, nc = 35, seq[k]
        #add_kmer(kmer_dict, key, idx, Nu, 35, seq[k], lastc=lastc, offbit=offbit)
        add_kmer(kmer_dict, key, empty, idx, Nu, hd, nc, lastc=lastc, offbit=offbit)

        shift = bit ** (k - 1)
        for i in xrange(k, n-1):
            cc = alpha[seq[i]]
            idx = i - k + 1
            Nu = Nu // bit + cc * shift
            # find head and next char
            hd, nc = seq[i-k], seq[i+1]

            add_kmer(kmer_dict, key, empty, idx, Nu, hd, nc, lastc=lastc, offbit=offbit)

        cc = alpha[seq[i+1]]
        Nu = Nu // bit + cc * shift
        hd, nc = seq[i-k], 36
        #add_kmer(kmer_dict, key, idx, Nu, hd, 36, lastc=lastc, offbit=offbit)
        add_kmer(kmer_dict, key, empty, idx, Nu, hd, nc, lastc=lastc, offbit=offbit)

    elif n == k:
        add_kmer(kmer_dict, key, empty, 0, k2n_jit_(seq), 35, 36, lastc=lastc, offbit=offbit)

    else:
        add_kmer(kmer_dict, key, empty, 0, -1, 35, 36, lastc=lastc, offbit=offbit)

    return 0

# jit version
build_dbg_jit_ = nb.njit(build_dbg)


# init my own hash table in jit
def init_dict(hashfunc=oakht, capacity=2**20, ksize=1, ktype=nb.uint64, vsize=1, vtype=nb.uint32, jit=True):
    if jit:
        spec = {}
        spec['capacity'] = nb.int64
        spec['load'] = nb.float32
        spec['size'] = nb.int64
        spec['ksize'] = nb.int64
        spec['vsize'] = nb.int64

        #spec['ktype'] = ktype
        #spec['keys'] = spec['ktype'][:]
        spec['keys'] = ktype[:]
        #spec['vtype'] = vtype
        #spec['values'] = spec['vtype'][:]
        spec['values'] = vtype[:]

        spec['counts'] = nb.uint8[:]
        clf = nb.jitclass(spec)(hashfunc)

    else:
        clf = hashfunc

    dct = clf(capacity=2**20, ksize=ksize, ktype=ktype, vsize=vsize, vtype=vtype)
    return dct
 

# build the dBG
#def seq2rdbg0(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**32, dump='breakpoint', saved='dBG_disk', hashfunc=oakht, jit=True, spec=spec):
def seq2rdbg_slow(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**63, dump='breakpoint', saved='dBG_disk', hashfunc=oakht, jit=True):

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
        kmer_dict = init_dict(hashfunc=oakht, capacity=2**20, ksize=1, ktype=nb.uint64, vtype=nb.uint16, jit=jit)

    seq_type = seq_chk(qry)

    N = 0
    f = open(qry, 'r')
    f.seek(breakpoint)
    flag = 0

    for i in SeqIO.parse(f, seq_type):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)

        #print('before adding node', i.id, len(seq_fw), 'clf size', kmer_dict.size, 'seq len', N)

        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            seq_bytes = seq.encode()
            res = build_dbg_jit_(seq_bytes, kmer_dict, kmer=kmer, bit=bits, offbit=offbit, lastc=lastc, alpha=alpha)
            N += n
            flag += n

        #print('after adding nodes', i.id, 'clf size', kmer_dict.size, 'seq len', N)

        # save the breakpoint
        if flag > chunk:
            bkt = rec_bkt(f, seq_type)
            print('breakpoint', bkt)
            _o = open(dump+'_seqs.txt', 'w')
            _o.write('%d'%bkt)
            _o.close()
            flag = 0
            kmer_dict.dump(dump)

        if N > Ns:
            break

    f.close()

    return kmer_dict

# parse sequences and save kmers in a dbg
@nb.njit
def seq2dbg_jit_0(seq_bytes, kmer_dict, isfasta, kmer, bits=5, offbit=offbit, lastc=lastc, alpha=alpha, N=0, Ns=2**63):
    #for qid, seq_fw in seqio_jit_(seq_bytes, isfasta=isfasta):
    for qid, seq_fw, ptr in seqio_jit_(seq_bytes, isfasta=isfasta):

        # build dbg from fwd
        res = build_dbg_jit_(seq_fw, kmer_dict, kmer=kmer, bit=bits, offbit=offbit, lastc=lastc, alpha=alpha)
        # build dbg from rev
        seq_rv = reverse_jit_(seq_fw)
        res = build_dbg_jit_(seq_rv, kmer_dict, kmer=kmer, bit=bits, offbit=offbit, lastc=lastc, alpha=alpha)
        N += 2 * len(seq_fw)

        if N > Ns:
            break

    return N

@nb.njit
#def seq2dbg_jit_(seq_bytes, kmer_dict, isfasta, kmer, bits=5, offbit=offbit, lastc=lastc, alpha=alpha, offset=0, chunk=2**32, N=0, Ns=2**63):
def seq2dbg_jit_(seq_bytes, kmer_dict, isfasta, kmer, bits=5, offbit=offbit, lastc=lastc, alpha=alpha, offset=0, chunk=2**32, Ns=2**63):
    N = 0
    #for qid, seq_fw in seqio_jit_(seq_bytes, isfasta=isfasta):
    chk = 0
    for qid, seq_fw, ptr in seqio_jit_(seq_bytes, offset=offset, isfasta=isfasta):

        # build dbg from fwd
        res = build_dbg_jit_(seq_fw, kmer_dict, kmer=kmer, bit=bits, offbit=offbit, lastc=lastc, alpha=alpha)
        # build dbg from rev
        seq_rv = reverse_jit_(seq_fw)
        res = build_dbg_jit_(seq_rv, kmer_dict, kmer=kmer, bit=bits, offbit=offbit, lastc=lastc, alpha=alpha)
        N += 2 * len(seq_fw)
        chk += 2 * len(seq_fw)
        if chk > chunk:
            return N, -1, ptr[0]

        if N > Ns:
            break

    return N, 1, ptr[0]


def seq2rdbg(qry, kmer=13, bits=5, Ns=1e6, chunk=2**32, brkpt='./breakpoint', saved='dBG_disk', hashfunc=oakht, jit=True):

    kmer = min(max(1, kmer), 27)
    size = int(pow(bits, kmer)+1)
    #breakpoint = 0
    if os.path.isfile(brkpt):
        offset, kmer_dict = load_on_disk(brkpt)

    else:
        kmer_dict = init_dict(hashfunc=oakht, capacity=2**20, ksize=1, ktype=nb.uint64, vsize=1, vtype=nb.uint16, jit=jit)
        offset = 0
    #seq_type = seq_chk(qry) 
    isfasta = seq_chk(qry) == 'fasta' and True or False

    seq_bytes = seq2bytes(qry)

    #N, done, ptr = seq2dbg_jit_(seq_bytes, kmer_dict, isfasta=isfasta, kmer=kmer, bits=bits, offbit=offbit, lastc=lastc, alpha=alpha, Ns=Ns)
    N = 0
    while 1:
        N_p, done, ptr = seq2dbg_jit_(seq_bytes, kmer_dict, isfasta=isfasta, kmer=kmer, bits=bits, offbit=offbit, lastc=lastc, alpha=alpha, offset=offset, chunk=chunk, Ns=Ns)
        if done == -1:
            offset = ptr
            # save the dbg on disk
            dump(kmer_dict, qry+'_db_brkpt', offset)
            gc.collect()
            print('1395 saving temp on disk', qry, kmer_dict.size, offset)
        else:
            break

        N += N_p
        if N > Ns:
            break

    return kmer_dict


# check if a kmer in dbg is also in rdbg, unused function, may be deprecated
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
def build_rdbg_jit_(rdbg_dict, kmer_dict):
    for kv in kmer_dict.iteritems():
        #k, hn = kv

        k, hns = kv
        hn = hns[0]

        pr = nbit_jit_(hn >> offbit)
        sf = nbit_jit_(hn & lowbit)
        if pr == sf == 1 and sf != 0b100000:
            continue
        else:
            #rdbg_dict.push(k, hn)
            rdbg_dict.push(k, hns)

    #return rdbg_dict
    return 0


# build rdbg from dbg
def dbg2rdbg(kmer_dict):
    rdbg_dict = init_dict(hashfunc=oakht, capacity=2**20, ksize=kmer_dict.ksize, ktype=nb.uint64, vsize=kmer_dict.vsize, vtype=nb.uint16, jit=True)
    res = build_rdbg_jit_(rdbg_dict, kmer_dict)
    return rdbg_dict


# get the weight of edge
@nb.njit
def rdbg_edge_weight(rdbg_edge, rdbg_dict, seq, kmer, bits=5):
    minus_one = nb.uint64(-1)
    path_cmpr = List.empty_list(nb.uint64)
    path_rdbg = List.empty_list(nb.uint64)
   
    idx, idx_prev = minus_one, minus_one
    for output in seq2ns_jit_(seq, kmer, bits):
        idx, k, hd, nt = output
        if k == minus_one:
            continue

        key = output[1:2]
        if rdbg_dict.has_key(key):
            path_rdbg.extend(output[1:4])

            if idx_prev == minus_one or idx_prev + kmer <= idx:
                path_cmpr.extend(output)
                idx_prev = idx

    # if path is empty, then sequence has non-repetitive k-mer, add the last k-mer to path
    if len(path_cmpr) == 0:
        path_cmpr.extend(output)

    N = len(path_rdbg)
    visit = {}
    for i in range(0, N-3, 3):
        n0, hd0, nt0 = path_rdbg[i: i+3]
        h0 = lastc[hd0] << offbit
        d0 = lastc[nt0]
 
        n1, hd1, nt1 = path_rdbg[i+3: i+6]
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

    del visit
    return 0


# compress the sequences into path and print
@nb.njit
def seq2path_jit_(seq, kmer, label_dct, bits, lastc=lastc, offbit=offbit):
    #starts = [0]
    starts = List()
    starts.append(0)

    #labels = [-1]
    labels = List()
    labels.append(-1)

    output = np.empty(3, nb.int32)

    for idx, k, hd, nt in seq2ns_jit_(seq, kmer, bits):

        #h = lastc[ord(hd)] << offbit
        h = lastc[hd] << offbit

        #d = lastc[ord(nt)]
        d = lastc[nt]

        kk = (nb.int64(k), nb.int64(h|d))


        if kk in label_dct:
            label = label_dct[kk]

            if starts[-1] < idx:
                pos = nb.int64(idx)+kmer
                #print('980 breakpoint', label, idx, kmer, pos) 

                # find different conserved region
                if labels[-1] != label:
                    labels.append(label)
                    starts.append(pos)
                # the same region, just extend it.
                else:
                    #starts[-1] = idx + kmer
                    starts[-1] = pos

        #raise SystemExit()

    for i1 in xrange(1, len(starts)):
        #print('%s\t%d\t%d\t%s\t%d'%(i.id, starts[idx-1], starts[idx], '+', labels[idx]))
        #print('%s\t%d\t%d\t%s\t%d'%(seqid, starts[idx-1], starts[idx], '+', labels[idx]))
        #yield starts[idx-1], starts[idx], labels[idx]

        output[0] = starts[i1-1]
        output[1] = starts[i1]
        output[2] = labels[i1]

        yield output

# convert sequences to paths and build the graph
#def seq2graph0(qry, kmer=13, bits=5, Ns=1e6, kmer_dict=None, saved=None, hashfunc=oakht, jit=True, spec=spec):
def seq2graph0(qry, kmer=13, bits=5, Ns=1e6, kmer_dict=None, saved=None, hashfunc=oakht, jit=True):

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
        load_dbg(saved, kmer_dict)

    #rdbg_dict = init_dict(hashfunc=oakht, capacity=2**20, ksize=kmer_dict.ksize, ktype=nb.uint64, vtype=nb.uint16, jit=jit)
    #rdbg_dict = build_rdbg(rdbg_dict, kmer_dict)
    rdbg_dict = build_rdbg(kmer_dict)

    #del kmer_dict.keys
    #del kmer_dict.values
    #del kmer_dict.counts
    #kmer_dict.clear()
    kmer_dict.destroy()
    del kmer_dict
    gc.collect()

    rdbg_edge = Dict()
    zero = nb.uint64(0)
    rdbg_edge[(zero, zero, zero, zero)] = 0
    N = 0

    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)
        for seq in [seq_fw]:
            seq_bytes = seq.encode()
            res = rdbg_edge_weight(rdbg_edge, rdbg_dict, seq_bytes, kmer, bits)
            n = len(seq)

        N += n
        if N > Ns:
            break

    #print('rdbg size 1665', len(rdbg_edge))
    _oname = qry + '_rdbg_weight.xyz'
    #_oname = './' +  _oname.split(os.sep)[-1]

    _o = open(_oname, 'w')
    for key in rdbg_edge:
        k12 = tuple(nb.uint64(elem) for elem in key)
        n0, hd0, n1, hd1 = k12
        val = rdbg_edge[k12]
        xyz = '%d_%d\t%d_%d\t%d\n'%(n0, hd0, n1, hd1, val)
        _o.write(xyz)

    _o.close()
    
    # call the mcl for clustering
    os.system('mcl %s --abc -I 1.5 -te 8 -o %s.mcl > log.mcl'%(_oname, _oname))

    del rdbg_edge
    del rdbg_dict
    gc.collect()
    
    label_dct = Dict()
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
        if kj not in label_dct:
            label_dct[kj] = flag
            flag += 1

        kk = tuple(map(int, k.split('_')[:2]))
        if kk not in label_dct:
            label_dct[kk] = flag
            flag += 1

    N = 0
    print('label_dct', len(label_dct))
    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)

        for seq in [seq_fw]:
            
            seq_bytes = seq.encode()
            for st, ed, lab in seq2path_jit_(seq_bytes, kmer, label_dct, bits=bits, lastc=lastc, offbit=offbit):
                print('%s\t%d\t%d\t%s\t%d'%(i.id, st, ed, '+', lab))

            #starts = [0]
            #labels = [-1]
            #for idx, k, hd, nt in seq2ns_(seq, kmer, bits):
            #
            #    if idx % 10**6 == 0:
            #        print('idx is', idx, len(seq))

            #    h = lastc[ord(hd)] << offbit
            #    d = lastc[ord(nt)]
            #    kk = (k, h|d)
 
            #    if kk in label_dct:
            #        label = label_dct[kk]
            #        if starts[-1] < idx:

            #            # find different conserved region
            #            if labels[-1] != label:
            #                labels.append(label)
            #                starts.append(idx+kmer)
            #            # the same region, just extend it.
            #            else:
            #                starts[-1] = idx + kmer

            #for idx in xrange(1, len(starts)):
            #    print('%s\t%d\t%d\t%s\t%d'%(i.id, starts[idx-1], starts[idx], '+', labels[idx]))

        N += len(seq_fw)
        if N > Ns:
            break

    return label_dct

# put sequences in the reduced dbg
def seq2graph_slow(qry, kmer=13, bits=5, Ns=1e6, rdbg_dict=None, saved=None, hashfunc=oakht, jit=True):

    kmer = min(max(1, kmer), 27)
    seq_type = seq_chk(qry)

    rdbg_edge = Dict()
    zero = nb.uint64(0)
    rdbg_edge[(zero, zero, zero, zero)] = 0

    N = 0
    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)
        for seq in [seq_fw]:
            seq_bytes = seq.encode()
            res = rdbg_edge_weight(rdbg_edge, rdbg_dict, seq_bytes, kmer, bits)
            n = len(seq)
    
        N += n
        if N > Ns:
            break

    #seq_bytes = seq2bytes(qry)
    #N = rdbg_edge_weight_jit_(rdbg_edge, rdbg_dict, seq_bytes, kmer, bits, Ns=Ns)

    #print('rdbg size 1665', len(rdbg_edge))
    _oname = qry + '_rdbg_weight.xyz'
    #_oname = './' +  _oname.split(os.sep)[-1]

    _o = open(_oname, 'w')
    for key in rdbg_edge:
        k12 = tuple(nb.uint64(elem) for elem in key)
        n0, hd0, n1, hd1 = k12
        val = rdbg_edge[k12]
        xyz = '%d_%d\t%d_%d\t%d\n'%(n0, hd0, n1, hd1, val)
        _o.write(xyz)

    _o.close()
    
    # call the mcl for clustering
    os.system('mcl %s --abc -I 1.5 -te 8 -o %s.mcl > log.mcl'%(_oname, _oname))

    del rdbg_edge
    del rdbg_dict
    gc.collect()
    
    label_dct = Dict()
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
        if kj not in label_dct:
            label_dct[kj] = flag
            flag += 1

        kk = tuple(map(int, k.split('_')[:2]))
        if kk not in label_dct:
            label_dct[kk] = flag
            flag += 1

    print('label_dct', len(label_dct))
    for i in SeqIO.parse(qry, seq_type):
        seq_fw = str(i.seq)
        for seq in [seq_fw]:
            seq_bytes = seq.encode()
            for st, ed, lab in seq2path_jit_(seq_bytes, kmer, label_dct, bits=bits, lastc=lastc, offbit=offbit):
                print('%s\t%d\t%d\t%s\t%d'%(i.id, st, ed, '+', lab))
    
        N += len(seq_fw)
        if N > Ns:
            break
    #for qid, st, ed, std, lab in seqs2path_jit_(seq_bytes, kmer, label_dct, bits=bits, lastc=lastc, offbit=offbit, Ns=2**63):
    #    qid_byte = bytes(qid).decode()[1:]
    #    strand = std == 1 and '+' or '-'
    #    print('%s\t%d\t%d\t%s\t%d'%(qid_byte, st, ed, strand, lab))

    return label_dct

# get the weight of the reduced dbg
@nb.njit
def rdbg_edge_weight_jit_(rdbg_edge, rdbg_dict, seq_bytes, isfasta, kmer, bits, N=0, Ns=2**63):
    #for qid, seq_fw in seqio_jit_(seq_bytes, isfasta=isfasta):
    for qid, seq_fw, ptr in seqio_jit_(seq_bytes, isfasta=isfasta):
        res = rdbg_edge_weight(rdbg_edge, rdbg_dict, seq_fw, kmer, bits)
        N += len(seq_fw)
        if N > Ns:
            break

    return N

# convert sequences into path
@nb.njit
def seqs2path_jit_(seq_bytes, isfasta, kmer, label_dct, bits=5, lastc=lastc, offbit=offbit, N=0, Ns=2**63):

    for qid, seq_fw, ptr in seqio_jit_(seq_bytes, isfasta):
        for st, ed, lab in seq2path_jit_(seq_fw, kmer, label_dct, bits=bits, lastc=lastc, offbit=offbit):
            #print('%s\t%d\t%d\t%s\t%d'%(i.id, st, ed, '+', lab))
            yield qid, st, ed, 1, lab

        N += len(seq_fw)
        if N > Ns:
            break


def seq2graph(qry, kmer=13, bits=5, Ns=1e6, rdbg_dict=None, saved=None, hashfunc=oakht, jit=True):

    kmer = min(max(1, kmer), 27)
    #seq_type = seq_chk(qry)
    isfasta = seq_chk(qry) == 'fasta' and True or False

    rdbg_edge = Dict()
    zero = nb.uint64(0)
    rdbg_edge[(zero, zero, zero, zero)] = 0

    seq_bytes = seq2bytes(qry)
    N = rdbg_edge_weight_jit_(rdbg_edge, rdbg_dict, seq_bytes, isfasta, kmer, bits, Ns=Ns)

    #print('rdbg size 1665', len(rdbg_edge))
    _oname = qry + '_rdbg_weight.xyz'
    #_oname = './' +  _oname.split(os.sep)[-1]

    _o = open(_oname, 'w')
    for key in rdbg_edge:
        k12 = tuple(nb.uint64(elem) for elem in key)
        n0, hd0, n1, hd1 = k12
        val = rdbg_edge[k12]
        xyz = '%d_%d\t%d_%d\t%d\n'%(n0, hd0, n1, hd1, val)
        _o.write(xyz)

    _o.close()
    
    # call the mcl for clustering
    os.system('mcl %s --abc -I 1.5 -te 8 -o %s.mcl > log.mcl'%(_oname, _oname))

    del rdbg_edge
    del rdbg_dict
    gc.collect()
    
    label_dct = Dict()
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
        if kj not in label_dct:
            label_dct[kj] = flag
            flag += 1

        kk = tuple(map(int, k.split('_')[:2]))
        if kk not in label_dct:
            label_dct[kk] = flag
            flag += 1

    print('label_dct', len(label_dct))
    for qid, st, ed, std, lab in seqs2path_jit_(seq_bytes, isfasta, kmer, label_dct, bits=bits, lastc=lastc, offbit=offbit, Ns=Ns):
        qid_byte = bytes(qid).decode()[1:]
        strand = std == 1 and '+' or '-'
        print('%s\t%d\t%d\t%s\t%d'%(qid_byte, st, ed, strand, lab))

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

    # test performance of sequence parse
    #try:
    #    fn = sys.argv[1]
    #    print('file name is', fn)
    #    seq_byte = seq2byte(fn)
    #    parse_test(seq_byte)
    #    print('finish sequences parse testing')
    #except:
    #    pass

    args = {'-i': '', '-k': '50', '-n': '2**63', '-r': '', '-d': ''}
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
            N = 10 ** 4

        mkey = 5
        print('test', N)

        pypy = platform.python_implementation().lower() 
        if pypy == 'pypy':
            clf = oakht(ksize=mkey)
        else:
            #oakht_jit = nb.jitclass(spec)(oakht)
            #clf = oakht_jit(capacity=int(N * 1.34), ksize=1, ktype=spec['ktype'], vtype=spec['vtype'])
            #clf = init_dict(ksize=mkey, ktype=nb.uint64, vtype=nb.uint32)
            clf = init_dict(hashfunc=oakht, capacity=2**20, ksize=mkey, ktype=nb.uint64, vtype=nb.uint16, jit=True)

        string = 'ATGC' * (N // 4)
        #seq = np.frombuffer(string.encode(), dtype=np.uint8)
        seq = np.frombuffer(string.encode(), dtype='uint8')
        np.random.shuffle(seq)

        print('initial finish', 'python version', pypy)
        #@nb.njit
        def oa_test(N, k, clf):
            x = np.random.randint(0, N, N)
            for i in range(N-k+1):
                clf.push(x[i:i+k], np.array([i]))

            flag = 0
            for i in range(N-k+1):
                if not clf.has_key(x[i:i+k]):
                    flag += 1

            print('x err', flag, 'ksize', k)
            flag = 0
            y = np.random.randint(0, N, N)
            for i in range(N-k+1):
                if clf.has_key(y[i:i+k]):
                    flag += 1

            print('y err', flag, 'ksize', k)
            flag = 0
            for kv in clf.iteritems():
                print('key is', kv[0], 'val is', kv[1])
                if flag > 5:
                    break
                flag += 1

        if pypy == 'pypy':
            print('pypy version')
            oa_test(N, mkey, clf)
        else:
            print('numba version')
            oa_test_jit = nb.jit(oa_test)
            oa_test_jit(N, mkey, clf)
        raise SystemExit()

    if dbs:
        #print('recover from', bkt)
        # convert sequence to path and build the graph
        dct = seq2graph(qry, kmer=kmer, bits=5, Ns=Ns, saved=dbs, hashfunc=oakht)
    else:
        # build the dbg
        print('# build the dBG')
        #kmer_dict = seq2rdbg(qry, kmer, 5, Ns, rec=bkt)
        kmer_dict = seq2rdbg(qry, kmer, 5, Ns, brkpt=bkt, chunk=2**33)
        #kmer_dict = seq2rdbg_slow(qry, kmer, 5, Ns, rec=bkt)
        #raise SystemExit()

        print('# save dBG to disk')
        dump(kmer_dict, qry+'_db')

        print('# load dBG from disk')
        del kmer_dict
        #kmer_dict = load_on_disk(qry+'_db.npz')
        offset, kmer_dict = load_on_disk(qry+'_db.npz')

        # convert dbg to reduced dbg
        print('# build the reduced dBG')
        rdbg_dict = dbg2rdbg(kmer_dict)
        kmer_dict.destroy()
        del kmer_dict
        gc.collect()

        # convert sequence to path
        print('# find fr')
        dct = seq2graph(qry, kmer=kmer, bits=5, Ns=Ns, rdbg_dict=rdbg_dict, hashfunc=oakht)
        #dct = seq2graph_slow(qry, kmer=kmer, bits=5, Ns=Ns, rdbg_dict=rdbg_dict, hashfunc=oakht)

    return 0
    #return dct


def target(*args):
    return entry_point, None

if __name__ == "__main__":

   entry_point(sys.argv)
