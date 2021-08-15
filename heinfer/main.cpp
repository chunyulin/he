#include "utils.h"
#include "heinfer.h"

int read_preprocessed(vector<vector<double>>& v, const char fname[]);

string parse_model_desc(const char fname[], vector<int>& s, vector<int>& p) {

    std::ifstream fin(fname, std::ios::in);
    if (!fin) {
        cout << "Cannot open " << fname<< endl;
        exit(0);
    }

    string line, model;
    fin >> s[0] >> s[1] >> s[2];    getline(fin, line);
    fin >> p[0] >> p[1] >> p[2];    getline(fin, line);
    getline(fin, line);  std::stringstream ss(line); 
    ss >> model;
    return model;
}


int main(int argc, char* argv[]) {

    TIMER t;
    usint nMults = 12; // max 2-depth tower
    usint depth = 3;   // max key for s^2
    //usint sf = 49, firstmod = 53;   // the maxmal setting for 32768 ringdim for 128-bit
    usint sf = 39, firstmod = 45;   // the maxmal setting for 32768 ringdim for 128-bit
    usint ringdim = 0;
    const char* DataFile  = argv[1];
    //const char* ModelFile = argv[2];
    const char* ModelDesc = argv[2];
    int ninf, nbp = 28500;
    usint pinf = 4096;

    if (argc == 1) {
        printf("Usage: %s <data.bin> <model.h5> [pinf=%d] [nbp=%d] [sf=%d] [decryBits=%d] [rD=%d] [nMults=%d] [depth=%d] \n", 
                       argv[0], pinf, nbp, sf, firstmod, ringdim, nMults, depth);
        exit(0);
    }

    if (argc > 3)   pinf    = atoi(argv[3]);
    if (argc > 4)   nbp    = atoi(argv[4]);
    if (argc > 5)   sf = atoi(argv[5]);
    if (argc > 6)   firstmod = atoi(argv[6]);
    if (argc > 7)   ringdim = atoi(argv[7]);
    if (argc > 8)   nMults = atoi(argv[8]);
    if (argc > 9)   depth = atoi(argv[9]);

    cout << "\n=========  HEInfer ( w/ packing factor ) =========" << endl;
    
    int ncore = omp_get_max_threads();
    cout << "# Core : " << ncore << endl << endl;


    TIC(t);
    cout << "Reading data file " << DataFile << " ... ";
    vector<vector<double>> fasta;
    int nbp_read = read_preprocessed(fasta, DataFile);
    ninf = fasta[0].size();
    
    DURATION tPre = TOC(t);
    cout << tPre.count() << " sec." << endl;
    cout << "   Read " << ninf << " data with each nbp = " << nbp_read << endl;
    if (nbp_read < nbp)   nbp = nbp_read;

    TIC(t);
    // Initialize HE context
    HEInfer he(ninf, nbp, nMults, depth, sf, firstmod, ringdim, pinf);
    he.genKeys();
    DURATION tKG = TOC(t);
    //he.test();


    printf("Reading model parameters ... ");
    vector<int> STRIDE(3);
    vector<int> POOLING(3);
    string ModelFile = parse_model_desc(ModelDesc, STRIDE, POOLING);
    cout << ModelFile << endl;
    printf("   Stride:  %d %d %d\n", STRIDE[0], STRIDE[1], STRIDE[2]);
    printf("   Pooling: %d %d %d\n", POOLING[0], POOLING[1], POOLING[2]);
    H5File h5d(ModelFile.c_str(), H5F_ACC_RDONLY);
    Layer conv1 (h5d, "/model_weights/conv1/conv1/kernel:0",   "/model_weights/conv1/conv1/bias:0");
    Layer conv2 (h5d, "/model_weights/conv2/conv2/kernel:0",   "/model_weights/conv2/conv2/bias:0");
    Layer conv3 (h5d, "/model_weights/conv3/conv3/kernel:0",   "/model_weights/conv3/conv3/bias:0");
    Layer output(h5d, "/model_weights/output/output/kernel:0", "/model_weights/output/output/bias:0");

    // pre-calculate output dimension of each layers
    int CH0 = conv1.wd[1];
    int in[4];
    in[0] = nbp;
    in[1] = conv1.getOutDim(in[0], STRIDE[0], POOLING[0]);
    in[2] = conv2.getOutDim(in[1], STRIDE[1], POOLING[1]);
    in[3] = conv3.getOutDim(in[2], STRIDE[2], POOLING[2]);  // output of conv3
    int nc[] = { he.calNCtxt( in[0] ),      // # ciphertext
                 he.calNCtxt( in[1]*conv1.wd[2] ),
                 he.calNCtxt( in[2]*conv2.wd[2] ),
                 he.calNCtxt( in[3]*conv3.wd[2] ),
                 output.wd[1]  };
    output.setInDim( in[3]*conv3.wd[2] );
    int nparam[] = { 0,   // # params
                     conv1.wd[2] * ( conv1.wd[0]*conv1.wd[1] + 1 ),
                     conv2.wd[2] * ( conv2.wd[0]*conv2.wd[1] + 1 ),
                     conv3.wd[2] * ( conv3.wd[0]*conv3.wd[1] + 1 ),
                     (in[3]*conv3.wd[2] + 1) * output.wd[1] };
    //output.setInDim(output.wd[0]);
    printf("Layer summary:  %d params\n", nparam[0]+nparam[1]+nparam[2]+nparam[3]+nparam[4]);
    printf("   In    - K:%5d x C:%2d = %5d  (NCtxt: %4d )\n", in[0], CH0, in[0]*CH0, nc[0]);
    printf("   Conv1 - K:%5d x C:%2d = %5d  (NCtxt: %4d, #param: %d)\n", in[1], conv1.wd[2], in[1]*conv1.wd[2], nc[1], nparam[1]);
    printf("   Conv2 - K:%5d x C:%2d = %5d  (NCtxt: %4d, #param: %d)\n", in[2], conv2.wd[2], in[2]*conv2.wd[2], nc[2], nparam[2]);
    printf("   Conv3 - K:%5d x C:%2d = %5d  (NCtxt: %4d, #param: %d)\n", in[3], conv3.wd[2], in[3]*conv3.wd[2], nc[3], nparam[3]);
    printf("   Dense -   %5d x %2d  (NCtxt: %4d, #param: %d)\n", in[3]*conv3.wd[2], output.wd[1], nc[4], nparam[4]);

    int n_ctx = nc[0]+nc[1]+nc[2]+nc[3];
    float total_size = (nMults + 1)* n_ctx * he.getRingDim() * 8.0 / 1024.0 / 1024.0 / 1024;
    printf("Estimated size of %d ciphertext: %.2f GB\n", n_ctx, total_size);

    // prepare ciphertext space
    CVec ctx0(nc[0]);     // input
    CVec ctx1(nc[1]);     // out of C1 layer 

    TIC(t);
    cout << "Encoding & Encrypting ... ";
    he.EncodeEncrypt(ctx0, fasta);
    DURATION tEnc = TOC(t);
    cout << tEnc.count() << " sec." << endl;
    cout << "   Dataset dimension: "<< fasta.size() << " x " << fasta[0].size() << endl;
    fasta.clear(); fasta.shrink_to_fit();


    //he.testDecrypt(ctx0[0]);

    cout << "Evaluating ..." << endl;
    TIC(t);
    he.ConvReluAP1d(ctx1, ctx0, conv1, STRIDE[0], POOLING[0]);
    DURATION tl1 = TOC(t);
    cout << "   Conv1... " << tl1.count() << " sec." << endl;
    
    //he.testDecrypt(ctx1[0]);

    ctx0.clear(); ctx0.shrink_to_fit();
    TIC(t);
    CVec ctx2(nc[2]);     // out of C3
    he.ConvReluAP1d(ctx2, ctx1, conv2, STRIDE[1], POOLING[1]);
    DURATION tl2 = TOC(t);
    cout << "   Conv2... " << tl2.count() << " sec." << endl;

    //he.testDecrypt(ctx2[0]);

    ctx1.clear(); ctx1.shrink_to_fit();
    TIC(t);
    CVec ctx3(nc[3]);     // out of C3
    he.ConvReluAP1d(ctx3, ctx2, conv3, STRIDE[2], POOLING[2]);
    DURATION tl3 = TOC(t);
    cout << "   Conv3... " << tl3.count() << " sec." << endl;

    //he.testDecrypt(ctx3[0]);

    ctx2.clear(); ctx2.shrink_to_fit();
    TIC(t);
    CVec ctx4(nc[4]);     // out of C3
    he.DenseSigmoid(ctx4, ctx3, output);
    DURATION tl4 = TOC(t);
    DURATION tEvalAll = tl1+tl2+tl3+tl4;
    cout << "   Dense... Total evaltion time : " << tEvalAll.count() << " sec." << endl;


    cout << "Decrypting... ";
    TIC(t);
    vector<vector<double>> prob = he.Decrypt(ctx4);
    DURATION tDec = TOC(t);
    cout << tDec.count() << " sec." << endl;

    //he.testDecrypt(ctx4[0]);

    // write prob
    std::ofstream outprob("prob.csv");
    outprob << "## Scores of label 1,2,3,4\n";
    for (int i=0; i<ninf; i++)
    for (int j=0; j<4; j++) {
        outprob << setprecision(8) << prob[j][i];
        if (j==3) outprob<< endl; else  outprob << ",";
    }
    outprob.close();

/*
    cout << "Scores for the first 4 item: " << endl;
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) cout << "   " << prob[j][i];
        cout << endl;
    }
*/

    double tmem = getMemoryUsage();
    printf("[Summary] Memory(MB): %.2f NBP: %d NCtxt: %d PF: %d RD: %d nMult: %d  Time for Reading Data: %f KeyGen: %f EncodeEncryption: %f Eval: %f Decrpt: %f\n", 
            tmem/1024.0, nbp, n_ctx, he.getPackingFactor(), he.getRingDim(), nMults,
            tPre.count(), tKG.count(), tEnc.count(), tEvalAll.count(), tDec.count());
    
    return 0;
}
