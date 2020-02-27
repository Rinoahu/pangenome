## Introduction
This is a graph-based method to find the frequency across species. This method includes several steps:

1. Build a dBG
2. Build the reduced dBG by removing nodes with <= 1 indegree and outdegree in dBG.
3. Convert sequence to compressed path according to 2.
4. Remove weak edges in rdBG and index the connect components in rdBG.
5. label each sequence again.

## Requirement

Make sure that you have the following installed

1. [Python](https://www.python.org/ "https://www.python.org/") (3.7 or greater) with numba ("http://numba.pydata.org/") installed. We strongly recommand Anaconda ("https://www.anaconda.com/")

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
    $pypy pangenome/kmer_pypy.py -m -i input.fasta -k 27 > result.tab


    -i: genome sequences in fasta format.

    -k: the kmer size. Currently, the kmer size is limited to 27, we will remove the limitation in the future.


## Result
The result is a tab-seperated file.

The 1st column is the sequence identifier.

The 2-4 columns are the start, end, and strand of the conversed region.

The 5th column is the index of the conserved region.

    For example:
    Chr1       0       3250    +       340
    Chr1       3250    6851    +       41
    Chr1       6851    7420    +       18344
    Chr1       7420    7661    +       25920
    Chr1       7661    7811    +       36243
    Chr1       7811    8015    +       15344
    Chr1       8015    8071    +       16029
    Chr1       8071    8105    +       35682
    Chr1       8105    9779    +       49500
    Chr1       9779    9806    +       7184



## Citation

To cite our work, please refer to:

xxx
