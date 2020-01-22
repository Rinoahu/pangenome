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
try:
    from _numpypy import multiarray as np
except:
    import numpy as np

try:
    xrange = xrange
except:
    xrange = range

def isprime(n) : 
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

# return prime list
def prime_list(n):
    start = int(math.log(n, 2))
    out = []
    for i in xrange(start, 63):
        N = 2**i
        for j in xrange(N, N+7*10**7):
            if isprime(j):
                out.append(j)
                break
    out.reverse()
    return out



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
        print(str(self[:]))

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
    return np.frombuffer(buf, dtype=dtype).reshape(shape), f

primes = [4611686018427388039, 2305843009213693967, 1152921504606847009, 576460752303423619,
        288230376151711813, 144115188075855881, 72057594037928017, 36028797018963971, 18014398509482143, 
        9007199254740997, 4503599627370517, 2251799813685269, 1125899906842679, 562949953421381, 
        281474976710677, 140737488355333, 70368744177679, 35184372088891, 17592186044423, 8796093022237, 
        4398046511119, 2199023255579, 1099511627791, 549755813911, 274877906951, 137438953481, 68719476767, 
        34359738421, 17179869209, 8589934609, 4294967311, 2147483659, 1073741827, 536870923, 268435459, 
        134217757, 67108879, 33554467, 16777259, 8388617, 4194319, 2097169, 1048583]

#print('primes', primes)
# open addressing hash table for kmer count based on robin hood algorithm
# has some bugs
class robin:
    def __init__(self, capacity=1024, load_factor = .85, key_type='uint64', val_type='uint16', disk=False):

        self.primes = [elem for elem in primes if elem > capacity]
        #self.primes = [elem for elem in primes if elem > capacity]
        self.capacity = self.primes.pop()
        # load factor is 2/3
        self.load = load_factor
        self.size = 0
        self.null = 2**64-1
        self.ktype = key_type
        self.vtype = val_type
        self.disk = disk
        self.radius = 0
        # for big, my own mmap based array can be used
        N = self.capacity

        # enable disk based hash
        if self.disk:
            self.keys, self.fk = memmap('tmp_key.npy', shape=N, dtype=key_type)
            self.values, self.fv = memmap('tmp_val.npy', shape=N, dtype=val_type)
            self.dist, self.fd = memmap('tmp_dis.npy', shape=N, dtype='uint8')

        else:
            self.keys = np.empty(N, dtype=key_type)
            self.values = np.empty(N, dtype=val_type)
            self.dist = np.empty(N, dtype='uint8')

        self.keys[:] = self.null
        self.dist[:] = 0


    def resize(self):
        N = self.capacity
        #M = N * 2
        M = self.primes.pop()
        #print('resize from %d to %d, size %d'%(N, M, self.size))
        null = self.null
        if self.disk:
            keys, fv = memmap('tmp_key0.npy', shape=M, dtype=self.ktype)
            values, fk = memmap('tmp_val0.npy', shape=M, dtype=self.vtype)
            dist, fd = memmap('tmp_dis0.npy', shape=M, dtype='uint8')

        else:
            keys = np.empty(M, dtype=self.ktype)
            values = np.empty(M, dtype=self.vtype)
            dist = np.empty(M, dtype='uint8')

        keys[:] = null
        dist[:] = 0

        self.capacity = M
        self.radius = 0
        # re-hash
        for i in xrange(N):
            key = int(self.keys[i])
            if key != null:
                value = self.values[i]
                # new hash
                j, k = hash(key) % M, 0
                j_init = j
                if key == 1093742941900681708:
                    print('before resize', j, key, M, j_init, keys[j_init])

                #j_rich, k_rich, diff = self.null, 255, 0
                #while key != keys[j] != null:
                #for k in xrange(N):
                for k in xrange(M):
                    #k += 1
                    #j = (j + 1) % M
                    j = (j_init + k) % M
                    #diff += 1
                    #if dist[j] < k_rich:
                    #    j_rich, k_rich, diff = j, dist[j], 0
                    if keys[j] == key or keys[j] == null:
                        break

                self.radius = max(k, self.radius)
                keys[j] = key
                values[j] = value
                dist[j] = k

                if key == 1093742941900681708:
                    print('after resize', j, keys[j], M, j_init, keys[j_init], k)

                #if k > k_rich:
                #    keys[j], keys[j_rich] = keys[j_rich], keys[j]
                #    values[j], values[j_rich] = values[j_rich], values[j]
                #    dist[j] = min(max(dist[j] - diff, 0), 255)
                #    dist[j_rich] = min(max(dist[j] + diff, 0), 255)

            else:
                continue

        if self.disk:
            # change name
            self.fk.close()
            self.fv.close()
            os.system('mv tmp_key0.npy tmp_key.npy && mv tmp_val0.npy tmp_val.npy && mv tmp_dis0.npy tmp_dis.npy')
            fk.close()
            fv.close()
            self.keys, self.fk = memmap('tmp_key.npy', 'a+', shape=M, dtype=self.ktype)
            self.values, self.fv = memmap('tmp_val.npy', 'a+', shape=M, dtype=self.vtype)
            self.dist, self.fd = memmap('tmp_dis.npy', 'a+', shape=M, dtype=self.vtype)

        else:
            self.keys = keys
            self.values = values
            self.dist = dist

        gc.collect()

    def pointer(self, key):
        key = int(key)
        M = self.capacity
        null = self.null
        j, k = hash(key) % M, 0
        j_init = j

        if key == 1093742941900681708:
            print('before point', j, M)

        # the rich point
        j_rich, k_rich, diff = null, 255, 0
        #while null != self.keys[j] != key:
        for k in xrange(M):
           #k += 1
            #j = (j + 1) % M
            j = (j_init + k) % M

            if self.dist[j] < k_rich:
                j_rich, k_rich, diff = j, self.dist[j], 0
            else:
                diff += 1

            if self.keys[j] == key or self.keys[j] == null:
                break

        if key == 1093742941900681708:
            print('after point', j, k, j_rich, k_rich, diff, M)

        self.radius = max(k, self.radius)
        return j, k, j_rich, k_rich, diff


    def __setitem__(self, key, value):
        j, k, j_rich, k_rich, diff = self.pointer(key)
        if self.keys[j] == self.null:
            self.size += 1
            self.keys[j] = key

        if key == 1093742941900681708 or (j_rich != self.null and self.keys[j_rich] == 1093742941900681708):
            print('after setting', j, self.keys[j], k, j_rich, self.keys[j_rich], k_rich, diff, self.capacity)

        self.values[j] = value
        self.dist[j] = k
        # swap
        if k > k_rich and j != j_rich:
        #if 0:
            if key == 1093742941900681708:
                print('after swap', j, k, j_rich, k_rich, diff, self.capacity)
            elif self.keys[j_rich] == 1093742941900681708:
                print('after swap fku', j, self.keys[j], k, j_rich, self.keys[j_rich], k_rich, diff, self.capacity)
            else:
                pass
            self.keys[j], self.keys[j_rich] = self.keys[j_rich], self.keys[j]
            self.values[j], self.values[j_rich] = self.values[j_rich], self.values[j]
            self.dist[j] = min(max(self.dist[j] - diff, 0), 255)
            self.dist[j_rich] = min(max(self.dist[j] + diff, 0), 255)

        # if too many elements
        if self.size * 1. / self.capacity > self.load:
            self.resize()
            #print('resize')

    def __getitem__(self, key):
        j, k, j_rich, k_rich, diff = self.pointer(key)
        #print('key', key, 'target', j, self.keys[j])
        if key == self.keys[j]:
            return self.values[j]
        else:
            raise KeyError

    def __delitem__(self, key):
        #j = self.pointer(key)
        j, k, j_rich, k_rich, diff = self.pointer(key)
        if key == self.keys[j]:
            self.keys[j] = self.null
            self.size -= 1
        else:
            raise KeyError

    def has_key(self, key):
        #j = self.pointer(key)
        j, k, j_rich, k_rich, diff = self.pointer(key)

        return key == self.keys[j] and True or False

    def __iter__(self):
        null = self.null
        for i in self.keys:
            if i != null:
                yield int(i)

    def __len__(self):
        return self.size


class mmapht:
    def __init__(self, size=1024, dtype='int16', fn='tmp'):
        self.values, self.fp = memmap(fn+'_values.npy', shape=size, dtype=dtype)
        self.counts, self.fc = memmap(fn+'_counts.npy', shape=size, dtype='uint8')
        self.fn = fn
        self.dtype = dtype

    def __getitem__(self, key):
        return self.values[key]

    def get_count(self, key):
        return self.counts[key]


    def __setitem__(self, key, value):
        self.values[key] = value
        count = self.counts[key]
        self.counts[key] = min(count+1, 255)

    def __iter__(self):
        for i in xrange(len(self.values)):
            yield i

    def __len__(self):
        return len(self.values)

    def dump(self, saved='dBG_disk'):
        self.fp.close()
        self.fc.close()
        os.system('mv %s_values.npy %s_values.npy'%(self.fn, saved))
        os.system('mv %s_counts.npy %s_counts.npy'%(self.fn, saved))

    def load(self, fn='dBG_disk'):
        self.fp.close()
        self.fc.close()
        os.system('rm %s_values.npy'%self.fn)
        os.system('rm %s_counts.npy'%self.fn)
        self.fn = self.fn
        self.values, self.fp = memmap(fn+'_values.npy', 'a+', shape=size, dtype=dtype)
        self.counts, self.fc = memmap(fn+'_counts.npy', 'a+', shape=size, dtype='uint8')
 

# open addressing hash table for kmer count
class oaht0:
    def __init__(self, capacity=1024, load_factor = .6666667, key_type='uint64', val_type='uint16', disk=False):

        self.primes = [elem for elem in primes if elem > capacity]
        #self.primes = [elem for elem in primes if elem > capacity]
        self.capacity = self.primes.pop()
        # load factor is 2/3
        self.load = load_factor
        self.size = 0
        self.null = 2**64-1
        self.ktype = key_type
        self.vtype = val_type
        self.disk = disk
        self.radius = 0
        # for big, my own mmap based array can be used
        N = self.capacity

        # enable disk based hash
        if self.disk:
            self.keys, self.fk = memmap('tmp_key.npy', shape=N, dtype=key_type)
            self.values, self.fv = memmap('tmp_val.npy', shape=N, dtype=val_type)
        else:
            self.keys = np.empty(N, dtype=key_type)
            self.values = np.empty(N, dtype=val_type)

        self.keys[:] = self.null


    def resize(self):
        N = self.capacity
        # re-hash
        if self.disk==False:
            # write old key and value to disk
            keys_old, fk_old = memmap('tmp_key_old.npy', shape=N, dtype=self.ktype)
            values_old, fv_old = memmap('tmp_val_old.npy', shape=N, dtype=self.vtype)
            keys_old[:] = self.keys
            values_old[:] = self.values
            fk_old.close()
            fv_old.close()
            keys_old, fk_old = memmap('tmp_key_old.npy', 'a+', shape=N, dtype=self.ktype)
            values_old,fv_old = memmap('tmp_val_old.npy', 'a+', shape=N, dtype=self.vtype)
            del self.keys, self.values
            gc.collect()
        else:
            keys_old, values_old = self.keys, self.values
 
        M = self.primes.pop()
        #print('resize from %d to %d, size %d'%(N, M, self.size))
        null = self.null
        if self.disk:
            keys, fk = memmap('tmp_key0.npy', shape=M, dtype=self.ktype)
            values, fv = memmap('tmp_val0.npy', shape=M, dtype=self.vtype)
        else:
            #print('extend array in ram')
            keys = np.empty(M, dtype=self.ktype)
            values = np.empty(M, dtype=self.vtype)

        keys[:] = null
        self.capacity = M
        self.radius = 0

        for i in xrange(N):
            key = keys_old[i]
            if key != null:
                value = values_old[i]
                # new hash
                j, k = hash(key) % M, 0
                j_init = j
                #while key != keys[j] != null:
                for k in xrange(N):
                #for k in itertools.count(0):
                    if keys[j] == key or keys[j] == null:
                        break

                    j = (j_init + k * k) % M
                    #k += 1
                    #mx_sum += 1

                self.radius = max(k, self.radius)
                keys[j] = key
                values[j] = value

                #if i % 10**5 == 0:
                #    print('resize iter', i, mx_sum)

            else:
                continue

        if self.disk:
            # change name
            self.fk.close()
            self.fv.close()
            os.system('mv tmp_key0.npy tmp_key.npy && mv tmp_val0.npy tmp_val.npy')
            fk.close()
            fv.close()
            self.keys, self.fk = memmap('tmp_key.npy', 'a+', shape=M, dtype=self.ktype)
            self.values, self.fv = memmap('tmp_val.npy', 'a+', shape=M, dtype=self.vtype)
        else:
            self.keys = keys
            self.values = values

        del keys_old, values_old

        if self.disk == False:
            fk_old.close()
            fv_old.close()
            os.system('rm tmp_key_old.npy tmp_val_old.npy')

        gc.collect()

    def resize0(self):
        N = self.capacity
        #M = N * 2
        M = self.primes.pop()
        #print('resize from %d to %d, size %d'%(N, M, self.size))
        null = self.null
        if self.disk:
            keys, fk = memmap('tmp_key0.npy', shape=M, dtype=self.ktype)
            values, fv = memmap('tmp_val0.npy', shape=M, dtype=self.vtype)
        else:
            #print('extend array in ram')
            keys = np.empty(M, dtype=self.ktype)
            values = np.empty(M, dtype=self.vtype)

        keys[:] = null
        self.capacity = M
        self.radius = 0

        # re-hash
        keys_old, values_old = self.keys, self.values

        for i in xrange(N):
            key = keys_old[i]
            if key != null:
                value = values_old[i]
                # new hash
                j, k = hash(key) % M, 0
                j_init = j
                #while key != keys[j] != null:
                for k in xrange(N):
                #for k in itertools.count(0):
                    if keys[j] == key or keys[j] == null:
                        break

                    j = (j_init + k * k) % M
                    #k += 1
                    #mx_sum += 1

                self.radius = max(k, self.radius)
                keys[j] = key
                values[j] = value

                #if i % 10**5 == 0:
                #    print('resize iter', i, mx_sum)

            else:
                continue

        if self.disk:
            # change name
            self.fk.close()
            self.fv.close()
            os.system('mv tmp_key0.npy tmp_key.npy && mv tmp_val0.npy tmp_val.npy')
            fk.close()
            fv.close()
            self.keys, self.fk = memmap('tmp_key.npy', 'a+', shape=M, dtype=self.ktype)
            self.values, self.fv = memmap('tmp_val.npy', 'a+', shape=M, dtype=self.vtype)
        else:
            self.keys = keys
            self.values = values

        del keys_old, values_old
        gc.collect()

    def pointer(self, key):
        M = self.capacity
        null = self.null
        j, k = hash(key) % M, 0
        j_init = j
        #while null != self.keys[j] != key:
        for k in xrange(M):
        #for k in itertools.count(0):
            if self.keys[j] == key or self.keys[j] == null:
                break

            #k += 1
            j = (j_init + k * k) % M

        self.radius = max(k, self.radius)

        return j


    def __setitem__(self, key, value):
        j = self.pointer(key)
        if self.keys[j] == self.null:
            self.size += 1
            self.keys[j] = key

        self.values[j] = value
        # if too many elements
        if self.size * 1. / self.capacity > self.load:
            self.resize()
            #print('resize')

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
                yield int(i)

    def __len__(self):
        return self.size


class oaht:
    def __init__(self, capacity=1024, load_factor = .6666667, key_type='uint64', val_type='uint16', disk=False):

        self.primes = [elem for elem in primes if elem > capacity]
        self.capacity = self.primes.pop()
        # load factor is 2/3
        self.load = load_factor
        self.size = 0
        self.null = 2**64-1
        self.ktype = key_type
        self.vtype = val_type
        self.disk = disk
        self.radius = 0
        # for big, my own mmap based array can be used
        N = self.capacity

        # enable disk based hash
        if self.disk:
            self.keys, self.fk = memmap('tmp_key.npy', shape=N, dtype=key_type)
            self.values, self.fv = memmap('tmp_val.npy', shape=N, dtype=val_type)
            self.counts, self.fc = memmap('tmp_cnt.npy', shape=N, dtype='uint8')

        else:
            self.keys = np.empty(N, dtype=key_type)
            self.values = np.empty(N, dtype=val_type)
            self.counts = np.empty(N, dtype='uint8')

        self.keys[:] = self.null


    def resize(self):
        N = self.capacity
        # re-hash
        if self.disk==False:
            # write old key and value to disk
            keys_old, fk_old = memmap('tmp_key_old.npy', shape=N, dtype=self.ktype)
            values_old, fv_old = memmap('tmp_val_old.npy', shape=N, dtype=self.vtype)
            counts_old, fc_old = memmap('tmp_cnt_old.npy', shape=N, dtype='uint8')

            keys_old[:] = self.keys
            values_old[:] = self.values
            counts_old[:] =  self.counts

            fk_old.close()
            fv_old.close()
            fc_old.close()

            keys_old, fk_old = memmap('tmp_key_old.npy', 'a+', shape=N, dtype=self.ktype)
            values_old,fv_old = memmap('tmp_val_old.npy', 'a+', shape=N, dtype=self.vtype)
            counts_old,fc_old = memmap('tmp_cnt_old.npy', 'a+', shape=N, dtype='uint8')

            del self.keys, self.values, self.counts
            gc.collect()

        else:
            keys_old, values_old, counts_old = self.keys, self.values, self.counts
 
        M = self.primes.pop()
        #print('resize from %d to %d, size %d'%(N, M, self.size))
        null = self.null
        if self.disk:
            keys, fk = memmap('tmp_key0.npy', shape=M, dtype=self.ktype)
            values, fv = memmap('tmp_val0.npy', shape=M, dtype=self.vtype)
            counts, fc = memmap('tmp_cnt0.npy', shape=M, dtype='uint8')

        else:
            #print('extend array in ram')
            keys = np.empty(M, dtype=self.ktype)
            values = np.empty(M, dtype=self.vtype)
            counts = np.empty(M, dtype='uint8')

        keys[:] = null
        self.capacity = M
        self.radius = 0

        for i in xrange(N):
            key = keys_old[i]
            if key != null:
                value = values_old[i]
                count = counts_old[i]
                # new hash
                j, k = hash(key) % M, 0
                j_init = j
                #while key != keys[j] != null:
                for k in xrange(N):
                #for k in itertools.count(0):
                    if keys[j] == key or keys[j] == null:
                        break

                    j = (j_init + k * k) % M
                    #k += 1
                    #mx_sum += 1

                self.radius = max(k, self.radius)
                keys[j] = key
                values[j] = value
                counts[j] = count

                #if i % 10**5 == 0:
                #    print('resize iter', i, mx_sum)

            else:
                continue

        if self.disk:
            # change name
            self.fk.close()
            self.fv.close()
            self.fc.close()

            os.system('mv tmp_key0.npy tmp_key.npy')
            os.system('mv tmp_val0.npy tmp_val.npy')
            os.system('mv tmp_cnt0.npy tmp_cnt.npy')

            fk.close()
            fv.close()
            fc.close()

            self.keys, self.fk = memmap('tmp_key.npy', 'a+', shape=M, dtype=self.ktype)
            self.values, self.fv = memmap('tmp_val.npy', 'a+', shape=M, dtype=self.vtype)
            self.counts, self.fc = memmap('tmp_cnt.npy', 'a+', shape=M, dtype='uint8')

        else:
            self.keys = keys
            self.values = values
            self.counts = counts

        del keys_old, values_old, counts_old

        if self.disk == False:
            fk_old.close()
            fv_old.close()
            fc_old.close()
            os.system('rm tmp_key_old.npy tmp_val_old.npy tmp_cnt_old.npy')

        gc.collect()

    def resize0(self):
        N = self.capacity
        #M = N * 2
        M = self.primes.pop()
        #print('resize from %d to %d, size %d'%(N, M, self.size))
        null = self.null
        if self.disk:
            keys, fk = memmap('tmp_key0.npy', shape=M, dtype=self.ktype)
            values, fv = memmap('tmp_val0.npy', shape=M, dtype=self.vtype)
        else:
            #print('extend array in ram')
            keys = np.empty(M, dtype=self.ktype)
            values = np.empty(M, dtype=self.vtype)

        keys[:] = null
        self.capacity = M
        self.radius = 0

        # re-hash
        keys_old, values_old = self.keys, self.values

        for i in xrange(N):
            key = keys_old[i]
            if key != null:
                value = values_old[i]
                # new hash
                j, k = hash(key) % M, 0
                j_init = j
                #while key != keys[j] != null:
                for k in xrange(N):
                #for k in itertools.count(0):
                    if keys[j] == key or keys[j] == null:
                        break

                    j = (j_init + k * k) % M
                    #k += 1
                    #mx_sum += 1

                self.radius = max(k, self.radius)
                keys[j] = key
                values[j] = value

                #if i % 10**5 == 0:
                #    print('resize iter', i, mx_sum)

            else:
                continue

        if self.disk:
            # change name
            self.fk.close()
            self.fv.close()
            os.system('mv tmp_key0.npy tmp_key.npy && mv tmp_val0.npy tmp_val.npy')
            fk.close()
            fv.close()
            self.keys, self.fk = memmap('tmp_key.npy', 'a+', shape=M, dtype=self.ktype)
            self.values, self.fv = memmap('tmp_val.npy', 'a+', shape=M, dtype=self.vtype)
        else:
            self.keys = keys
            self.values = values

        del keys_old, values_old
        gc.collect()

    def pointer(self, key):
        M = self.capacity
        null = self.null
        j, k = hash(key) % M, 0
        j_init = j
        #while null != self.keys[j] != key:
        for k in xrange(M):
        #for k in itertools.count(0):
            if self.keys[j] == key or self.keys[j] == null:
                break

            #k += 1
            j = (j_init + k * k) % M

        self.radius = max(k, self.radius)

        return j


    def __setitem__(self, key, value):
        j = self.pointer(key)
        if self.keys[j] == self.null:
            self.size += 1
            self.keys[j] = key
            self.counts[j] = 0

        self.values[j] = value
        count = self.counts[j]
        self.counts[j] = min(count + 1, 255)

        # if too many elements
        if self.size * 1. / self.capacity > self.load:
            self.resize()
            #print('resize')

    def __getitem__(self, key):
        j = self.pointer(key)
        #print('key', key, 'target', j, self.keys[j])
        if key == self.keys[j]:
            return self.values[j]
        else:
            raise KeyError

    def get_count(self, key):
        j = self.pointer(key)
        if key == self.keys[j]:
            return self.counts[j]
        else:
            return 0


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
                yield int(i)

    def __len__(self):
        return self.size

    # save hash table to disk
    def dump(self, fname):
        N = len(self.keys)
        key_type, val_type = self.ktype, self.vtype
        dump_keys, dump_fk = memmap(fname + '_dump_key.npy', shape=N, dtype=key_type)
        dump_keys[:] = self.keys
        dump_fk.close()

        dump_values, dump_fv = memmap(fname + '_dump_val.npy', shape=N, dtype=val_type)
        dump_values[:] = self.values
        dump_fv.close()

        dump_counts, dump_fc = memmap(fname + '_dump_cnt.npy', shape=N, dtype='uint8')
        dump_counts[:] = self.counts
        dump_fc.close()

    # load hash table from disk
    def loading(self, fname):
        key_type, val_type = self.ktype, self.vtype

        #dump_keys, dump_fk = memmap(fname + '_dump_key.npy', 'a+', shape=N, dtype=key_type)
        dump_keys, dump_fk = memmap(fname + '_dump_key.npy', 'a+', dtype=key_type)

        self.keys = np.array(dump_keys)
        dump_fk.close()

        dump_values, dump_fv = memmap(fname + '_dump_val.npy', 'a+', dtype=val_type)
        self.values = np.array(dump_values)
        dump_fv.close()

        dump_counts, dump_fc = memmap(fname + '_dump_cnt.npy', 'a+', dtype='uint8')
        self.counts = np.array(dump_counts)
        dump_fc.close()
        capacity  = len(self.counts)
        self.primes = [elem for elem in primes if elem >= capacity]
        self.capacity = self.primes.pop()

        print('loading length', map(len, [dump_keys, dump_values, dump_counts]))

        self.size = sum(self.keys != self.null)


# support multiple key
class oamkht:
    def __init__(self, capacity=1024, load_factor = .75, mkey=1, key_type='uint64', val_type='uint16', disk=False):

        self.primes = [elem for elem in primes if elem > capacity]
        self.capacity = self.primes.pop()
        # load factor is 2/3
        self.load = load_factor
        self.size = 0
        self.null = 2**64-1
        self.ktype = key_type
        self.vtype = val_type
        self.disk = disk
        self.radius = 0
        self.mkey = mkey
        # for big, my own mmap based array can be used
        N = self.capacity

        # enable disk based hash
        if self.disk:
            self.keys, self.fk = memmap('tmp_key.npy', shape=mkey*N, dtype=key_type)
            self.values, self.fv = memmap('tmp_val.npy', shape=N, dtype=val_type)
            self.counts, self.fc = memmap('tmp_cnt.npy', shape=N, dtype='uint8')

        else:
            self.keys = np.empty(mkey * N, dtype=key_type)
            self.values = np.empty(N, dtype=val_type)
            self.counts = np.empty(N, dtype='uint8')

        self.keys[::mkey] = self.null


    # whether key0 == key1
    def eq(self, k0, s0, k1, s1, N):
        if N == 1:
            return k0[s0] == k1
        else:
            for i in xrange(N):
                if k0[i+s0] != k1[i+s1]:
                    return False
            return True


    def resize(self):
        N = self.capacity
        mkey = self.mkey
        # re-hash
        if self.disk==False:
            # write old key and value to disk
            keys_old, fk_old = memmap('tmp_key_old.npy', shape=mkey * N, dtype=self.ktype)
            values_old, fv_old = memmap('tmp_val_old.npy', shape=N, dtype=self.vtype)
            counts_old, fc_old = memmap('tmp_cnt_old.npy', shape=N, dtype='uint8')

            #iprint(keys_old.shape, self.keys.shape)

            keys_old[:] = self.keys
            values_old[:] = self.values
            counts_old[:] =  self.counts

            fk_old.close()
            fv_old.close()
            fc_old.close()

            keys_old, fk_old = memmap('tmp_key_old.npy', 'a+', shape=mkey * N, dtype=self.ktype)
            values_old,fv_old = memmap('tmp_val_old.npy', 'a+', shape=N, dtype=self.vtype)
            counts_old,fc_old = memmap('tmp_cnt_old.npy', 'a+', shape=N, dtype='uint8')

            del self.keys, self.values, self.counts
            gc.collect()

        else:
            keys_old, values_old, counts_old = self.keys, self.values, self.counts
 
        M = self.primes.pop()
        #print('resize from %d to %d, size %d'%(N, M, self.size))
        null = self.null
        if self.disk:
            keys, fk = memmap('tmp_key0.npy', shape=mkey * M, dtype=self.ktype)
            values, fv = memmap('tmp_val0.npy', shape=M, dtype=self.vtype)
            counts, fc = memmap('tmp_cnt0.npy', shape=M, dtype='uint8')

        else:
            #print('extend array in ram')
            keys = np.empty(mkey * M, dtype=self.ktype)
            values = np.empty(M, dtype=self.vtype)
            counts = np.empty(M, dtype='uint8')

        keys[::mkey] = null
        self.capacity = M
        self.radius = 0

        for i in xrange(N):
            #key = keys_old[:, i]
            im = i * mkey

            if mkey > 1:
                key = tuple(keys_old[im: im+mkey])
                key0 = key[0]
            else:
                key0 = key = keys_old[im]

            if key0 != null:
                value = values_old[i]
                count = counts_old[i]
                # new hash
                j, k = hash(key) % M, 0

                j_init = j
                #while key != keys[j] != null:
                for k in xrange(N):
                #for k in itertools.count(0):
                    jm = j * mkey
                    #if keys[jm] == null or all(keys[jm:jm+mkey] == key) :
                    if keys[jm] == null or self.eq(keys, jm, key, 0, mkey) :
                        break

                    j = (j_init + k * k) % M
                    #k += 1
                    #mx_sum += 1

                self.radius = max(k, self.radius)
                #keys[:, j] = key
                jm = j*mkey
                keys[jm: jm+mkey] = key
                #print('resize', jm, M, keys[jm], key)
                values[j] = value
                counts[j] = count

                #if i % 10**5 == 0:
                #    print('resize iter', i, mx_sum)

            else:
                continue

        if self.disk:
            # change name
            self.fk.close()
            self.fv.close()
            self.fc.close()

            os.system('mv tmp_key0.npy tmp_key.npy')
            os.system('mv tmp_val0.npy tmp_val.npy')
            os.system('mv tmp_cnt0.npy tmp_cnt.npy')

            fk.close()
            fv.close()
            fc.close()

            self.keys, self.fk = memmap('tmp_key.npy', 'a+', shape=mkey*M, dtype=self.ktype)
            self.values, self.fv = memmap('tmp_val.npy', 'a+', shape=M, dtype=self.vtype)
            self.counts, self.fc = memmap('tmp_cnt.npy', 'a+', shape=M, dtype='uint8')

        else:
            self.keys = keys
            self.values = values
            self.counts = counts

        del keys_old, values_old, counts_old

        if self.disk == False:
            fk_old.close()
            fv_old.close()
            fc_old.close()
            os.system('rm tmp_key_old.npy tmp_val_old.npy tmp_cnt_old.npy')

        gc.collect()

    def pointer(self, key):
        mkey = self.mkey
        M = self.capacity
        null = self.null

        if mkey > 1:
            key = tuple(key)

        j, k = hash(key) % M, 0
        k = 0
        j_init = j
        #while null != self.keys[j] != key:
        for k in xrange(M):
        #for k in itertools.count(0):
            jm = j * mkey
            #if all(self.keys[jm:jm+mkey] == key) or self.keys[jm] == null:
            if self.eq(self.keys, jm, key, 0, mkey) or self.keys[jm] == null:
                break

            #k += 1
            j = (j_init + k * k) % M

        self.radius = max(k, self.radius)

        return j


    def __setitem__(self, key, value):
        j = self.pointer(key)
        mkey = self.mkey
        jm = mkey * j
        if self.keys[jm] == self.null:
            self.size += 1
            #print('before set', self.keys[jm: jm+mkey], key)
            self.keys[jm:jm+mkey] = key
            #print('after set', self.keys[jm: jm+mkey], key)
            self.counts[j] = 0

        self.values[j] = value
        count = self.counts[j]
        self.counts[j] = min(count + 1, 255)

        # if too many elements
        if self.size * 1. / self.capacity > self.load:
            self.resize()
            #print('resize')

    def __getitem__(self, key):
        j = self.pointer(key)
        #print('key', key, 'target', j, self.keys[j])
        mkey = self.mkey
        jm = j * mkey
        #if all(key == self.keys[jm: jm+mkey]):
        #if self.eq(key, 0, self.keys, jm, mkey):
        if self.eq(self.keys, jm, key, 0, mkey):
            return self.values[j]
        else:
            raise KeyError

    def get_count(self, key):
        j = self.pointer(key)
        mkey = self.mkey
        jm = j * mkey
        #if all(key == self.keys[jm: jm+mkey]):
        #print(key, 0, self.keys[:2], jm, mkey)
        #if self.eq(key, 0, self.keys, jm, mkey):
        if self.eq(self.keys, jm, key, 0, mkey):
            return self.counts[j]
        else:
            return 0


    def __delitem__(self, key):
        j = self.pointer(key)
        mkey = self.mkey
        jm = j * mkey
        #if all(key == self.keys[jm: jm+mkey]):
        #if self.eq(key, 0, self.keys, jm, mkey):
        if self.eq(self.keys, jm, key, 0, mkey):
            self.keys[jm] = self.null
            self.size -= 1
        else:
            raise KeyError

    def has_key(self, key):
        j = self.pointer(key)
        mkey = self.mkey
        jm = j * mkey
        #return all(key == self.keys[jm:jm+mkey])
        #return self.eq(key, 0, self.keys, jm, mkey)
        #print('has key', jm, self.capacity, self.keys[jm], key)
        return self.eq(self.keys, jm, key, 0, mkey)
   

    def __iter__(self):
        null = self.null
        #for i in self.keys:
        #    if i != null:
        #        yield int(i)
        mkey = self.mkey
        if mkey > 1:
            for i in xrange(0, self.keys.shape[0], mkey):
                if self.keys[i] != null:
                    yield self.keys[i:i+mkey]
        else:
            for i in xrange(0, self.keys.shape[0], mkey):
                if self.keys[i] != null:
                    yield self.keys[i]
 
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

        self.mkey = len(self.keys) // len(self.values)

        capacity  = len(self.counts)
        self.primes = [elem for elem in primes if elem >= capacity]
        self.capacity = self.primes.pop()

        #print('loading length', map(len, [dump_keys, dump_values, dump_counts]))
        self.size = sum(self.keys[:: self.mkey] != self.null)

# combine several dict
class mdict:
    def __init__(self, capacities=1024, load_factor=.75, key_type='uint64', val_type='uint16', fuc=oaht, bucket=32):
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


def seq2ns_(seq, k=12, bit=5):
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

# bisect based query
# xs is the sorted array
# x is the query
def query0(xs, x):
    idx = bisect_left(xs, x)
    return x in xs[idx:idx+1]

# check node exist
def query(ht, i):
    try:
        hn = ht[i]
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
        #print('seq', seq_fw[:10], seq_rv[:10])

        #itr = 0
        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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
        path = []
        #print('seq', seq_fw)
        #for seq in [seq_fw, seq_rv]:
        for seq in [seq_fw]:
            skip = p0 = p1 = 0
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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
        for seq in [seq_fw, seq_rv][:1]:
            n = len(seq)
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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
        path = []
        for seq in [seq_fw]:
            skip = p0 = p1 = 0
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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
def seq2dbg2(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**32, dump='breakpoint', saved='dBG_disk', hashfunc=oamkht):
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
    #for i in SeqIO.parse(qry, seq_type):
    for i in SeqIO.parse(f, seq_type):
        seq_fw = str(i.seq)
        seq_rv = str(i.reverse_complement().seq)
        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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
        path = []
        for seq in [seq_fw]:
            skip = p0 = p1 = 0
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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
def seq2dbg(qry, kmer=13, bits=5, Ns=1e6, rec=None, chunk=2**32, dump='breakpoint', saved='dBG_disk', hashfunc=oamkht):
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
        for seq in [seq_fw, seq_rv]:
            n = len(seq)
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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
        for seq in [seq_fw]:
            path_cmpr = []
            path_rdbg = []
            skip = p0 = p1 = 0
            for k, hd, nt in seq2ns_(seq, kmer, bits):
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



# print the manual
def manual_print():
    print('Usage:')
    print('  pyhton this.py -i qry.fsa -k 10 -n 1000000')
    print('Parameters:')
    print('  -i: query sequences in fasta format')
    print('  -k: kmer length')
    print('  -n: length of query sequences for dbg')

def entry_point(argv):

    args = {'-i': '', '-k': '50', '-n': '1000000', '-r': ''}
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
    qry, kmer, Ns, bkt = args['-i'], int(args['-k']), int(eval(args['-n'])), args['-r']
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
        from random import randint
        N = 10**6
        mkey = 5
        clf = oamkht(mkey=mkey)

        if mkey>1:
            x = [tuple([randint(0, N) for tmp in range(mkey)]) for elem in xrange(N)]
            y = [tuple([randint(0, N) for tmp in range(mkey)]) for elem in xrange(N)]
        else:
            x = [randint(0, 2**63) for elem in xrange(N)]
            y = [randint(0, 2**63) for elem in xrange(N)]

        for i in x:
            try:
                val = min(255, i[0])
            except:
                val = min(255, i)
            clf[i] = min(255, val)

        flag = 0
        for i in x:
            try:
                val = min(255, i[0])
            except:
                val = min(255, i)

            if clf.has_key(i):
                flag += 1
            #else:
            #    print('not eq', i, val, i in x)

        print('x == %d'%len(x), flag)

        flag = 0
        for i in y:
            if clf.has_key(i):
                flag += 1
        print('y == 0', flag)
        raise SystemExit()

    print('recover from', bkt)
    dct = seq2dbg(qry, kmer, 5, Ns, rec=bkt)
    return 0
    #return dct


def target(*args):
    return entry_point, None

if __name__ == "__main__":

    entry_point(sys.argv)
