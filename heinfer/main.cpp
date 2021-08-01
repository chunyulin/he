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
    int ninf=2000, nbp = 28500;
    usint pinf = 0;

    if (argc == 1) {
        printf("Usage: %s <h5 file> [nbp=%d] [pinf=%d] [sf=%d] [decryBits=%d] [rD=%d] [nMults=%d] [depth=%d] \n", 
                       argv[0], nbp, pinf, sf, firstmod, ringdim, nMults, depth);
        exit(0);
    }

    if (argc > 2)   nbp    = atoi(argv[2]);
    if (argc > 3)   pinf    = atoi(argv[3]);
    if (argc > 4)   sf = atoi(argv[4]);
    if (argc > 5)   firstmod = atoi(argv[5]);
    if (argc > 6)   ringdim = atoi(argv[6]);
    if (argc > 7)   nMults = atoi(argv[7]);
    if (argc > 8)   depth = atoi(argv[8]);
    
    HEInfer he(ninf, nbp, nMults, depth, sf, firstmod, ringdim, pinf);
    he.genKeys();
    //he.test();

    printf("Reading model parameters ...\n");
    H5File h5d(H5NAME, H5F_ACC_RDONLY);
    Layer conv1 (h5d, "/model_weights/conv1/conv1/kernel:0",   "/model_weights/conv1/conv1/bias:0");
    Layer conv2 (h5d, "/model_weights/conv2/conv2/kernel:0",   "/model_weights/conv2/conv2/bias:0");
    Layer conv3 (h5d, "/model_weights/conv3/conv3/kernel:0",   "/model_weights/conv3/conv3/bias:0");
    Layer output(h5d, "/model_weights/output/output/kernel:0", "/model_weights/output/output/bias:0");
    const vector<int> STRIDE {6,4,2};
    const vector<int> POOLING{2,2,2};

    // pre-calculate output dimension of each layers
    int CH0 = conv1.wd[1];
    int in[4];
    in[0] = nbp;
    in[1] = conv1.getOutDim(in[0], STRIDE[0], POOLING[0]);
    in[2] = conv2.getOutDim(in[1], STRIDE[1], POOLING[1]);
    in[3] = conv3.getOutDim(in[2], STRIDE[2], POOLING[2]);  // output of conv3
    int nc[] = { he.calNCtxt( in[0] ), 
                 he.calNCtxt( in[1]*conv1.wd[2] ),
                 he.calNCtxt( in[2]*conv2.wd[2] ),
                 he.calNCtxt( in[3]*conv3.wd[2] ),
                 output.wd[1]  };
    output.setInDim( in[3]*conv3.wd[2] );
    //output.setInDim(output.wd[0]);
    printf("Layer summary : \n");
    printf("      In    - K:%d\t x C:%d\t = %d\t  (NCtxt: %d)\n", in[0], CH0, in[0]*CH0, nc[0]);
    printf("  C1: Conv1 - K:%d\t x C:%d\t = %d\t  (NCtxt: %d)\n", in[1], conv1.wd[2], in[1]*conv1.wd[2], nc[1]);
    printf("  C2: Conv2 - K:%d\t x C:%d\t = %d\t  (NCtxt: %d)\n", in[2], conv2.wd[2], in[2]*conv2.wd[2], nc[2]);
    printf("  C3: Conv3 - K:%d\t x C:%d\t = %d\t  (NCtxt: %d)\n", in[3], conv3.wd[2], in[3]*conv3.wd[2], nc[3]);
    printf("  D1: Dense - %d x %d                 (NCtxt: %d)\n", in[3]*conv3.wd[2], output.wd[1], nc[4]);

    int n_ctx = nc[0]+nc[1]+nc[2]+nc[3];
    float total_size = (nMults + 1)* n_ctx * he.getRingDim() * 8.0 / 1024.0 / 1024.0 / 1024;
    printf("Estimated size of %d ciphertext: %.2f GB\n", n_ctx, total_size);

    // prepare ciphertext space
    CVec ctx0(nc[0]);     // input
    CVec ctx1(nc[1]);     // out of C1 layer 

    TIC(t)
    cout << "Preprocessing " << FASTA_FILE << " ...\t";
    vector<int> exact(ninf);
    vector<vector<double>> fasta(nbp, vector<double>(ninf));
    preprocess_fasta(fasta, exact, FASTA_FILE);
    DURATION tPre = TOC(t);
    cout << tPre.count() << " sec." << endl;

    TIC(t)
    cout << "Encoding & Encrypting ...\t";
    he.EncodeEncrypt(ctx0, fasta);
    DURATION tEnc = TOC(t);
    cout << tEnc.count() << " sec." << endl;
    //fasta.clear();

    he.testDecrypt(ctx0[0]);

    cout << "Evalating ..." << endl;
    TIC(t);
    he.ConvReluAP1d(ctx1, ctx0, conv1, STRIDE[0], POOLING[0]);
    DURATION tl = TOC(t);
    cout << "  C1... " << tl.count() << " sec." << endl;
    
    he.testDecrypt(ctx1[0]);

    //ctx0.clear();
    CVec ctx2(nc[2]);     // out of C3
    he.ConvReluAP1d(ctx2, ctx1, conv2, STRIDE[1], POOLING[1]);
    tl = TOC(t);
    cout << "  C2... " << tl.count() << " sec." << endl;

    he.testDecrypt(ctx2[0]);

    //ctx1.clear();
    CVec ctx3(nc[3]);     // out of C3
    he.ConvReluAP1d(ctx3, ctx2, conv3, STRIDE[2], POOLING[2]);
    tl = TOC(t);
    cout << "  C3... " << tl.count() << " sec." << endl;

    he.testDecrypt(ctx3[0]);

    //ctx2.clear();
    CVec ctx4(nc[4]);     // out of C3
    he.DenseSigmoid(ctx4, ctx3, output);
    DURATION tEvalAll = TOC(t);
    cout << "  D1... Total evaltion time : " << tEvalAll.count() << " sec." << endl;


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
    printf("Max resident memory: %.2f GB, %.2f MB per ciphertext, %.2f MB in theorem. ( NBP: %d, NCtxt: %d, PF: %d, RD: %d  )\n", 
            tmem/1024.0/1024.0, tmem/1024.0/n_ctx, (nMults + 1)* he.getRingDim() * 8.0 / 1024.0 / 1024,
            nbp, n_ctx, he.getPackingFactor(), ringdim);
    
    return 0;
}
