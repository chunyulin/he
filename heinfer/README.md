## Compile

make preprocess
make -j


## Preprocess:

```
./preprocess <binary output name> <fasta file name> [NBP=28500] [kmer=3] [stride=3]

e.g., ./preprocess test.bin /data/test.fa
```


## CNN Classifier

```
./main test.bin /data/CNN634_s641.best.h5 4096
```
The "packing unit" the multiple of 2048 that not exceed the slots.
