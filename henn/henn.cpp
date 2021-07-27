#include "layer.h"
#include "utils.h"
#include <omp.h>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cassert>

using std::setw;
using std::setprecision;

// Headers for Palisade ------------------------------------------
#include <palisade/pke/palisade.h>
using namespace lbcrypto;
typedef vector<Ciphertext<DCRTPoly>> CVec;

// Headers  for timing ------------------------------------------
#include <chrono>
typedef  std::chrono::duration<double> DURATION;
std::chrono::time_point<std::chrono::system_clock>  tc, tw, t, t1;
#undef TIC
#undef TOC
#define TIC(t) t = std::chrono::system_clock::now();
#define TOC(t) (std::chrono::system_clock::now() - t)

CryptoContext<DCRTPoly> cc;

//#define AUTORESCALE
#include "ccTest.h"


/**
 *  Conv1D:
 *  Y_{I,f} = Pool[ \sigma ( X_{S*I+k,c}*W_{k,c,f} + B_{f} ) ]       for each {I, f}
 */
void heConvReluAP1d(CVec& out, const CVec& in, Layer& l, const int S=1, const int P=2) {

    cout << "Conv-Relu-Pooling layer:  in " << in.size() << "  out " << out.size() << endl;

    const int K=l.wd[0];
    const int C=l.wd[1];
    const int F=l.wd[2];
    const int I=l.getOutDim();
    cout << "  KCF: " << K << " " << C << " " << F << " " << I << endl;

    // compensate rescaling model parameters for average pooling
    #pragma omp parallel for
    for (int i=0; i<K*C*F; i++) l.w[i] *= 1.0/sqrt(P);
    #pragma omp parallel for
    for (int i=0; i<F; i++)     l.b[i] *= 1.0/sqrt(P);

    #pragma omp parallel for collapse(2)
    for (int i=0; i<I; i++)
    for (int f=0; f<F; f++) {
        CVec x(P), x2(P);

        for (int p=0; p<P; p++) {
            x[p] = cc->EvalAdd( cc->EvalMult(in[(i*S+p)*C], l.w[f]), l.b[f] );   // part for k = c =0
    
            for (int c=1; c<C; c++)
                x[p] += cc->EvalMult( in[(i*S+p)*C+1], l.w[F+f] );               // part for k=0, c=1
    
            for (int k=1; k<K; k++)
            for (int c=0; c<C; c++)                                              // the rest parts
                x[p] += cc->EvalMult( in[(i*S+k+p)*C+c], l.w[(k*C+c)*F+f] );

            x2[p] = cc->EvalMultNoRelin(x[p], x[p]);
        }

        // ReLU [x^2], followed by average pooling
        for (int p=1; p<P; p++) cc->EvalAddInPlace(x2[0], x2[p]);

        #ifdef AUTORESCALE
        out[i*F+f] = cc->Relinearize( x2[0] );
        #else
        cc->RelinearizeInPlace( x2[0] );
        cc->ModReduceInPlace( x2[0] );    // Totally rescale 3 times.
        cc->ModReduceInPlace( x2[0] );    // Can be reduce to 2-times if moving rescale into the loop-p, 
        out[i*F+f] = cc->ModReduce( x2[0] );
        #endif
    }

}


/**
 *  Dense:
 *  Y_i = X_j*W_ji * + B_i
 */
void heDense(CVec& out, const CVec& in, Layer& l, int s=1) {
    const int J=l.wd[0];
    const int I=l.wd[1];
    cout << "Dense layer: " << J << " x " << I << endl;

    const double c0 = 0.5;
    const double c1 = 0.25;
    const double c3 = - 1.0/48;


    #pragma omp parallel for
    for (int i=0; i<I; i++) {

        auto x = cc->EvalAdd( cc->EvalMult( in[0], l.w[i] ), l.b[i] ); // do j=0 first..
        for (int j=1; j<J; j++) {
            x += cc->EvalMult( in[j], l.w[j*I+i] );
        }

        #ifdef SIGMOID_ORD_3
        
        // Another expression in [-8,8]:  0.5 - 1.2/8 * x + 0.81562/8 * x**3
        // Taylor expansion Sigmoid: 0.5 + 0.25*x - x**3/48
        #ifdef AUTORESCALE
        auto x2   = cc->EvalMultNoRelin(x, x);
        auto xo48 = cc->EvalMult(x, c3);
        auto x3   = cc->EvalMultNoRelin(x2, xo48);

        auto xo4  = cc->EvalMult(x, c1);
        cc->EvalAddInPlace(x3, xo4);
        out[i] = cc->EvalAdd(x3, c0);
        #else
        auto x2   = cc->EvalMultNoRelin(x, x);    // d2, c3
        cc->ModReduceInPlace( x2 );
        cc->ModReduceInPlace( x2 );
        auto xo48 = cc->EvalMult(x, c3);
        auto x3   = cc->EvalMultNoRelin(x2, xo48);
        cc->ModReduceInPlace(x3);
        cc->ModReduceInPlace(x3);

        auto xo4  = cc->EvalMult(x, c1);
        cc->EvalAddInPlace(x3, xo4);
        cc->ModReduceInPlace(x3);
        out[i] = cc->EvalAdd(x3, c0);
        #endif
        
        #else
        
        const double c1 = 0.15625;
        
        
        #endif


    }

#if 0
//// Binary-tree addition....
//0.5 + 0.25*x - x**3/48
//    for(int j = 1; j < nbp; j=j*2)
//    #pragma omp parallel for
//    for(int i = 0; i < nbp; i = i + 2*j) {
//       if ( (i+j) < nbp )  cc->EvalAddInPlace(work[i], work[i+j]);
//    }
#endif

}

void ccInfo(const CryptoContext<DCRTPoly>& cc) {

    auto ccParam = cc->GetCryptoParameters();
    cout << "[ CryptoContext Info ]" << endl;
    cout << "  Rdim / Cyclo   : " << cc->GetRingDimension() << " " <<  cc->GetCyclotomicOrder() << endl;
    //cout << "  GetRootOfUnity : " << cc->GetRootOfUnity() << endl;
    cout << "  ScaleFactor = " << ccParam->GetPlaintextModulus() << endl;
    
    auto paramsQ = cc->GetElementParams()->GetParams();
    cout << "  Moduli in Q: " << endl;
    for (uint32_t i = 0; i < paramsQ.size(); i++) {
       auto qi = paramsQ[i]->GetModulus();
       cout << "    q" << i << ": " << qi.GetLengthForBase(2) << " - " << qi << endl;
    }
    
    auto ccParamCKKS = std::static_pointer_cast<LPCryptoParametersCKKS<DCRTPoly>>(ccParam);
    auto paramsQP = ccParamCKKS->GetParamsQP();
    cout << "  Moduli in P: " << endl;
    BigInteger P = BigInteger(1);
    for (uint32_t i = 0; i < paramsQP->GetParams().size(); i++) {
        if (i > paramsQ.size()) {
            P = P * BigInteger(paramsQP->GetParams()[i]->GetModulus());
            cout << "    p" << i - paramsQ.size() << ": "
                 << paramsQP->GetParams()[i]->GetModulus() << endl;
        }
    }
    auto QBitLength = cc->GetModulus().GetLengthForBase(2);
    auto PBitLength = P.GetLengthForBase(2);
    cout << "  Q bit length: " << QBitLength << " - " << cc->GetModulus() << endl;
    cout << "  P bit length: " << PBitLength << " - " << P << endl;
    cout << "  Total bit of ciphertext modulus: " << QBitLength + PBitLength << "\n\n";
}

int main(int argc, char* argv[]) {

    if (argc == 1) {
        cout << "Usage: ./henn <h5 file> [nMults] [mdepth] [sf] [decryBits] [rD]" << endl;
        exit(0);
    }
    const char* H5NAME = argv[1];


    int ncore = omp_get_max_threads();
         
    usint nbp = 28500;   //29000;    // # of neuclotides
    usint batchSize = 2048;


    SecurityLevel securityLevel = HEStd_128_classic;
    usint nMults = 15;    // max 2-depth tower
    usint maxdepth = 4;  // max key for s^2
    usint scaleFactor = 30; // will also effect ring dimention
    usint firstModSize = 40;
    usint numLargeDigits = 0;
    usint relinWindow = 0;   /* 0 means using CRT */
    int ringDimension = 0;   // default 8192 for std_128, but also depend on sf

    if (argc > 2)   nMults = atoi(argv[2]);
    if (argc > 3)   maxdepth = atoi(argv[3]);
    if (argc > 4)   scaleFactor = atoi(argv[4]);
    if (argc > 5)   firstModSize = atoi(argv[5]);
    if (argc > 6)   numLargeDigits = atoi(argv[6]);
    if (argc > 7)   ringDimension = atoi(argv[7]);

    cout << "===== HeNN =====" << endl;
    cout << "Core: " << ncore << endl;
    cout << "Weight file: " << H5NAME << endl;
    printf("Numbers BP : %d\n", nbp);
    printf("Batch      : %d\n\n", batchSize);
    printf("nMults=%d, Depth=%d, sf=%d, fb=%d\n", nMults, maxdepth, scaleFactor, firstModSize);



    cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, scaleFactor, batchSize, securityLevel, ringDimension,
        #ifdef AUTORESCALE
            APPROXAUTO, HYBRID, //(Rd=16384),
        #else
            APPROXRESCALE, HYBRID, //(Rd=16384),
        #endif
            //APPROXRESCALE, BV(Rd=8192),
            numLargeDigits, maxdepth, firstModSize, relinWindow, OPTIMIZED);

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);   // Not for BFV

    ccInfo(cc);
    
    int nbatch = 1;
    cout << "len(x), batchSize, # batch : " << nbp << " " << batchSize << " " << nbatch << endl;

    TIC(t1);
    cout << "Generate pub/sec keys... ";
    LPKeyPair<DCRTPoly> keyPair = cc->KeyGen();
    DURATION tKG = TOC(t1);
    cout << tKG.count() << " sec." << endl;

    TIC(t1);
    cout << "Generate multi keys... ";
    cc->EvalMultKeyGen(keyPair.secretKey);
    DURATION tMultKG = TOC(t1);
    cout << tMultKG.count() << " sec." << endl;

    #ifdef VERBOSE_HE
    //========================
    // Some unit test ...
    //ccTestSigmoid(keyPair);  exit(0);
    //ccTestDepth(keyPair, nMults, maxdepth);  exit(0);
    //ccTest(keyPair);  exit(0);
    #endif

    //
    // read model parameter
    //
    H5File h5d(H5NAME, H5F_ACC_RDONLY);
    Layer conv1 (h5d, "/model_weights/conv1/conv1/kernel:0",   "/model_weights/conv1/conv1/bias:0");
    Layer conv2 (h5d, "/model_weights/conv2/conv2/kernel:0",   "/model_weights/conv2/conv2/bias:0");
    Layer conv3 (h5d, "/model_weights/conv3/conv3/kernel:0",   "/model_weights/conv3/conv3/bias:0");
    Layer output(h5d, "/model_weights/output/output/kernel:0", "/model_weights/output/output/bias:0");
    const vector<int> STRIDE {6,4,2};
    const vector<int> POOLING{2,2,2};

    // pre-calculate output dimension of each layers
    int in0 = nbp;
    int CH0 = conv1.wd[1];
    int in1 = conv1.getOutDim(in0, STRIDE[0], POOLING[0]);
    int in2 = conv2.getOutDim(in1, STRIDE[1], POOLING[1]);
    int in3 = conv3.getOutDim(in2, STRIDE[2], POOLING[2]);
    output.setInDim(output.wd[0]);
    cout << "Conv-Relu-Pooling dim : " << endl;
    cout << "  " << in0 << " x " << CH0         << " = " << in0*CH0         << endl;
    cout << "  " << in1 << " x " << conv1.wd[2] << " = " << in1*conv1.wd[2] << endl;
    cout << "  " << in2 << " x " << conv2.wd[2] << " = " << in2*conv2.wd[2] << endl;
    cout << "  " << in3 << " x " << conv3.wd[2] << " = " << in3*conv3.wd[2] << endl;

    // prepare ciphertext space
    const int nprob = output.wd[1];
    
    CVec ctx0(in0*CH0);            // input   of 1st      layer
    CVec ctx1(in1*conv1.wd[2]);    // out(in) of 1st(2nd) layer 


    // Estimate print size of ciphertext
    int n_ctx = ctx0.size() + ctx1.size();
    float total_size = (nMults + 1)* n_ctx * cc->GetRingDimension() * 8.0 / 1024.0 / 1024.0 / 1024;
    printf("Estimated size of %d ciphertext: %.2f GB\n", n_ctx, total_size);


    cout << "Encoded/Encrypt data ... ";
    vector<double> x(batchSize);
    #pragma omp parallel for
    for (int i=0; i<batchSize; i++) x[i] = (double) (2*rand()- RAND_MAX) / (RAND_MAX);

    TIC(t);
    #pragma omp parallel for
    for (int i=0; i<nbp*CH0; i++) {
      if (i%1000==0) cout << " " << i;
      Plaintext ptx = cc->MakeCKKSPackedPlaintext(x);
      ctx0[i] = cc->Encrypt(keyPair.publicKey, ptx);
    }
    
    DURATION tEncAll = TOC(t);
    cout <<   " " << tEncAll.count() << " sec." << endl;

    #ifdef VERBOSE_HE
    testDecrypt(keyPair, ctx0[5]);
    #endif

    cout << "Eval..." << endl;
    TIC(t);

    heConvReluAP1d(ctx1, ctx0, conv1, STRIDE[0], POOLING[0]);
    DURATION tl = TOC(t);
    cout << "  C1... " << tl.count() << " sec." << endl;
    
    #ifdef VERBOSE_HE
    testDecrypt(keyPair, ctx1[0]);
    #endif

    delete &ctx0;
    CVec ctx2(in2*conv2.wd[2]);
    heConvReluAP1d(ctx2, ctx1, conv2, STRIDE[1], POOLING[1]);
    tl = TOC(t);
    cout << "  C2... " << tl.count() << " sec." << endl;

    #ifdef VERBOSE_HE
    testDecrypt(keyPair, ctx2[0]);
    #endif

    delete &ctx1;
    CVec ctx3(in3*conv3.wd[2]);    // output of flatten
    heConvReluAP1d(ctx3, ctx2, conv3, STRIDE[2], POOLING[2]);
    tl = TOC(t);
    cout << "  C3... " << tl.count() << " sec." << endl;

    #ifdef VERBOSE_HE
    testDecrypt(keyPair, ctx3[0]);
    #endif

    delete &ctx2;
    CVec ctx4(nprob);    // output
    heDense(ctx4, ctx3, output);
    tl = TOC(t);
    cout << "  D1... " << tl.count() << " sec." << endl;

    DURATION tEvalAll = TOC(t);

    cout << "Decrypt... ";
    TIC(t);
    vector<Plaintext> prob(nprob);
    #pragma omp parallel for
    for (int i=0; i<nprob; i++) {
        cc->Decrypt(keyPair.secretKey, ctx4[i], &prob[i]);
        prob[i]->SetLength(5);
    }
    DURATION tDec = TOC(t);

    cout <<   "  LevelDepth: " << prob[0]->GetLevel() << " " << prob[0]->GetDepth() << " "
                               << ctx4[0]->GetElements().size() << prob[0] << endl;


    cout << "=== Timing (s):" << endl;
    cout << "Pub/Sec Gen  : " << tKG.count() <<endl;
    cout << "MultKey Gen  : " << tMultKG.count() <<endl;
    cout << "Pack/Enc /np : " << tEncAll.count()/nbp <<endl;
    cout << "Eval     /np : " << tEvalAll.count()/nbp <<endl;
    cout << "Decrypt      : " << tDec.count() <<endl;

    printMemoryUsage();
    return 0;
}
