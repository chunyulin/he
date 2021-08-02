#include "utils.h"
#include "heinfer.h"


const char FASTA_FILE[] = "/work/lincy/iDASH2021/Challenge/small.fa";

void preprocess_fasta(vector<vector<double>>&, vector<int>&, const char[]);


int main(int argc, char* argv[]) {

    int ncore = omp_get_max_threads();
    cout << "# Core : " << ncore << endl << endl;

    TIMER t;
    
    usint nMults = 17; // max 2-depth tower
    usint depth = 3;   // max key for s^2
    usint sf = 35, firstmod = 46;   // the maxmal setting for 32768 ringdim for 128-bit
    usint ringdim = 0;
    const char* H5NAME = argv[1];
    //int ninf=2000, nbp = 28500;
    int ninf=2000, nbp = 16384;

    if (argc == 1) {
        printf("Usage: %s <h5 file> [nbp=%d] [sf=%d] [decryBits=%d] [rD=%d] [nMults=%d] [depth=%d] \n", 
                       argv[0], nbp, sf, firstmod, ringdim, nMults, depth);
        exit(0);
    }

    if (argc > 2)   nbp    = atoi(argv[2]);
    if (argc > 3)   sf = atoi(argv[3]);
    if (argc > 4)   firstmod = atoi(argv[4]);
    if (argc > 5)   ringdim = atoi(argv[5]);
    if (argc > 6)   nMults = atoi(argv[6]);
    if (argc > 7)   depth = atoi(argv[7]);

    HEInfer he(ninf, nbp, nMults, depth, sf, firstmod, ringdim);

    if (nbp > he.getSlots()) {
        cout << "Reduce nbp to " << he.getSlots() << endl;
        nbp = he.getSlots();
    }

    printf("Reading model parameters ...\n");
    H5File h5d(H5NAME, H5F_ACC_RDONLY);
    Layer conv1 (h5d, "/model_weights/conv1/conv1/kernel:0",   "/model_weights/conv1/conv1/bias:0");
    Layer conv2 (h5d, "/model_weights/conv2/conv2/kernel:0",   "/model_weights/conv2/conv2/bias:0");
    Layer conv3 (h5d, "/model_weights/conv3/conv3/kernel:0",   "/model_weights/conv3/conv3/bias:0");
    Layer output(h5d, "/model_weights/output/output/kernel:0", "/model_weights/output/output/bias:0");
    const vector<int> STRIDE {6,4,2};
    const vector<int> POOLING{2,2,2};
    vector<int> sp{ 1 };
    sp.push_back(sp[0]*POOLING[0]*STRIDE[0]);
    sp.push_back(sp[1]*POOLING[1]*STRIDE[1]);
    sp.push_back(sp[2]*POOLING[2]*STRIDE[2]);

    // pre-calculate output dimension of each layers
    int CH0 = conv1.wd[1];
    int nc[] = { ninf * CH0,
                 ninf * conv1.wd[2],
                 ninf * conv2.wd[2],
                 ninf * conv3.wd[2],
                 ninf * output.wd[1]   };
    int in[4];
    in[0] = nbp;
    in[1] = conv1.getOutDim(in[0], STRIDE[0], POOLING[0]);
    in[2] = conv2.getOutDim(in[1], STRIDE[1], POOLING[1]);
    in[3] = conv3.getOutDim(in[2], STRIDE[2], POOLING[2]);  // output of conv3
    output.setInDim( in[3]*conv3.wd[2] );
    printf("Layer summary : \n");
    printf("  In    - K:%d\t x C:%d\t = %d\t  (NCtxt: %d)\n", in[0], CH0, in[0]*CH0, nc[0]);
    printf("  Conv1 - K:%d\t x C:%d\t = %d\t  (NCtxt: %d)\n", in[1], conv1.wd[2], in[1]*conv1.wd[2], nc[1]);
    printf("  Conv2 - K:%d\t x C:%d\t = %d\t  (NCtxt: %d)\n", in[2], conv2.wd[2], in[2]*conv2.wd[2], nc[2]);
    printf("  Conv3 - K:%d\t x C:%d\t = %d\t  (NCtxt: %d)\n", in[3], conv3.wd[2], in[3]*conv3.wd[2], nc[3]);
    printf("  Dense - %d x %d                 (NCtxt: %d)\n", in[3]*conv3.wd[2], output.wd[1], nc[4]);

    // Prepare rotation keys
    vector<int> rot_idx;
    int nr = 0;
    for (int i=1; i<conv1.wd[0]; i++, nr++)  rot_idx.push_back(i*sp[0]);
    for (int i=1; i<conv2.wd[0]; i++, nr++)  rot_idx.push_back(i*sp[1]);
    for (int i=1; i<conv3.wd[0]; i++, nr++)  rot_idx.push_back(i*sp[2]);
    he.genKeys(rot_idx);

    //he.test();

    int n_ctx = nc[0]+nc[1]+nc[2]+nc[3];
    float total_size = (nMults + 1)* n_ctx * he.getRingDim() * 8.0 / 1024.0 / 1024.0 / 1024;
    printf("Estimated size of %d ciphertext: %.2f GB\n", n_ctx, total_size);

    // prepare ciphertext space
    CVec ctx0(nc[0]);     // input
    CVec ctx1(nc[1]);     // out of C1 layer 

    TIC(t)
    cout << "Preprocessing " << FASTA_FILE << " ...\t";
    vector<int> exact(ninf);
    vector<vector<double>> fasta(ninf, vector<double>(nbp));
    preprocess_fasta(fasta, exact, FASTA_FILE);
    DURATION tPre = TOC(t);
    cout << tPre.count() << " sec." << endl;

    TIC(t)
    cout << "Encoding & Encrypting ...\t";
    he.EncodeEncrypt(ctx0, fasta);
    DURATION tEnc = TOC(t);
    cout << tEnc.count() << " sec." << endl;
    fasta.clear();

    he.testDecrypt(ctx0[0]);

    cout << "Evalating ..." << endl;
    TIC(t);
    he.ConvReluAP1d(ctx1, ctx0, conv1, STRIDE[0], POOLING[0], sp[0]);
    DURATION tl = TOC(t);
    printf("  Conv1(%d) ... %s sec.", sp[0], tl.count() );

    he.testDecrypt(ctx1[0]);

    ctx0.clear();
    CVec ctx2(nc[2]);
    he.ConvReluAP1d(ctx2, ctx1, conv2, STRIDE[1], POOLING[1], sp[1]);
    tl = TOC(t);
    printf("  Conv2(%d) ... %s sec.", sp[1], tl.count() );

    he.testDecrypt(ctx2[0]);

    ctx1.clear();
    CVec ctx3(nc[3]);
    he.ConvReluAP1d(ctx3, ctx2, conv3, STRIDE[2], POOLING[2], sp[2]);
    tl = TOC(t);
    printf("  Conv3(%d) ... %s sec.", sp[2], tl.count() );

    he.testDecrypt(ctx3[0]);

    ctx2.clear();
    CVec ctx4(nc[4]);     // out of C3
    he.DenseSigmoid(ctx4, ctx3, output,  conv3.wd[2], sp[3]);
    DURATION tEvalAll = TOC(t);
    printf("  Dense(%d) ... %s sec.", sp[3], tEvalAll.count() );


    cout << "Decrypting... ";
    TIC(t);
    vector<vector<double>> prob = he.Decrypt(ctx4);
    DURATION tDec = TOC(t);
    cout << tDec.count() << " sec." << endl;

    he.testDecrypt(ctx4[0]);

    // write prob
    std::ofstream outprob("prob.txt");
    outprob << "## probability for label 1,2,3,4\n";
    for (int i=0; i<ninf; i++) {
        for (int j=0; j<4; j++) outprob << prob[j][i] << "\t"; 
        outprob << endl; 
    }
    outprob.close();

    
    double tmem = getMemoryUsage();
    printf("[Summary] Memory/MB: %f NBP: %d NCtxt: %d RD: %d nMult: %d Time: %f %f %f %f\n", 
            tmem/1024.0, nbp, n_ctx, he.getRingDim(), nMults,
            tPre.count(), tEnc.count(), tEvalAll.count(), tDec.count());
    
    return 0;
}
