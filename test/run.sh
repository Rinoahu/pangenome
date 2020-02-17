#!/bin/bash

tstime='/usr/bin/time -v'
#pypy=/home/xiaohu/gnu_tools/pypy/bin/pypy
pypy=python

seq=../../pan_genome.fsa
#seq=../../test.fsa
#seq=test.fsa

$tstime $pypy ../kmer_numba.py -i $seq -k 27 -n 1e7 &> log.txt

exit 0

jellyfish count -m 16 -s 100M -t 8 -o kmer_count -c 7 $seq
jellyfish dump -c -t -L 1 -U 1000000 kmer_count | grep \AAGTCATATCGCCA > kmer_count.txt
echo "####" >> kmer_count.txt
grep \AAGTCATATCGCCA log.txt >> kmer_count.txt
