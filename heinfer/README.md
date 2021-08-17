## Compile

make
make preprocess


## Preprocess:

```
./preprocess <output.bin> <fasta file> [readlabel=0] [SEGLEN=28500] [offset=200] [ninf=2000] [kmer=3] [stride=3]

e.g., ./preprocess test.bin /data/test.fa
```


## CNN Classifier
```
./main test.bin model.1053
```
with a Keras-trained model in HDF5 format and a model descriptor.
