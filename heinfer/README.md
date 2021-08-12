## Compile

make
make preprocess


## Preprocess:

```
./preprocess <output.bin> <fasta file> [readlabel=0] [SEGLEN=28500] [ninf=2000] [kmer=3] [offset=3]

e.g., ./preprocess test.bin /data/test.fa
```


## CNN Classifier

```
./main test.bin /data/CNN634_s641.best.h5
```
