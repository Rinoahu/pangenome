## Introduction
This is a graph-based method to find the frequency across species. This method includes several steps:

1. Build a dBG
2. Build reduced dBG by removing nodes with <= 1 indegree and outdegree in dBG
3. Convert sequence to compressed path according to 2
4. Remove weak edges in dBG and label each connect component of the rdBG
5. label each sequence again

## Requirement

Make sure that you have the following installed

1. [PyPy2.7](http://pypy.org/download.html "http://pypy.org/download.html") (v5.10 or greater).

2. [MCL](https://micans.org/mcl "https://micans.org/mcl"), a Markov Clustering algorithm.


## Download

    $git clone https://github.com/Rinoahu/pangenome

<!--## Install and Test

    $cd SwiftOrtho
    $bash ./install.sh
    $cd example
    $bash ./run.sh
-->


## Usage
$pypy pangenome/kmer_pypy.py -m -i input.fasta -k 27 > output

-i: genome sequences in fasta format.
-k: the kmer size.


## Result
The result is a tab-seperated file.
The 1st column is the sequence identifier
The 2nd and 3rd columns are the start and end posistion
The 4th column is the index of the conserved region.


## Citation

To cite our work, please refer to:

xxx
