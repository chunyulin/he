## Preprocessing:

./preprocess <binary output name> <fasta file name> [NBP=28500] [kmer=3] [stride=3]

Example:
    ./preprocess data.bin ../data/Testing.fa
    
    
## HE Classification with CNN:
    ./main data.bin CNN_fs8.h5
    