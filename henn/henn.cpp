#include "layer.h"
#include "utils.h"
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iomanip>
using std::setw;
using std::setprecision;

// Headers for Palisade
#include <palisade/pke/palisade.h>
using namespace lbcrypto;
typedef vector<Ciphertext<DCRTPoly>> CVec;

// Headers  for timing
#include <chrono>
typedef  std::chrono::duration<double> DURATION;
std::chrono::time_point<std::chrono::system_clock>  tc, tw, t, t1;
#undef TIC
#undef TOC
#define TIC(t) t = std::chrono::system_clock::now();
#define TOC(t) (std::chrono::system_clock::now() - t)

CryptoContext<DCRTPoly> cc;

void heActSigmoid(Ciphertext<DCRTPoly>& in) {
//0.5 + 0.25*x - x**3/48
//    for(int j = 1; j < nbp; j=j*2)
//    #pragma omp parallel for
//    for(int i = 0; i < nbp; i = i + 2*j) {
//       if ( (i+j) < nbp )  cc->EvalAddInPlace(work[i], work[i+j]);
//    }
}

void heActRelu(Ciphertext<DCRTPoly>& in) {
    // 0.119782 + 0.5*x + 0.147298 * x*x - 0.002015* x*x*x*x
    in = cc->EvalMult(in, in);
}


/**
 *  Conv1D:
 *  Y_{I,f} = X_{S*I+k,c}*W_{k,c,f} + B_{f}    for each {I, f}
 */
void heConv1d(CVec& out, const CVec& in, Layer& l, int S=1) {
    int K=l.wd[0];
    int C=l.wd[1];
    int F=l.wd[2];
    int I=l.getOutDim();
    
    cout << "Conv: " << K << " " << C << " " << F << " " << I << endl;

    #pragma omp parallel for collapse(2)
    for (int i=0; i<I; i++)
    for (int f=0; f<F; f++) {
    
      out[i*F+f] = cc->EvalAdd( cc->EvalMult(in[i*S*C], l.w[f]), l.b[f] );

      for (int c=1; c<C; c++) {
        out[i*F+f] += cc->EvalMult( in[i*S*C+c], l.w[(c)*F+f] );
      }
      for (int k=1; k<K; k++)
      for (int c=0; c<C; c++) {
        out[i*F+f] += cc->EvalMult( in[i*S*C+c], l.w[(k*C+c)*F+f] );
      }
      //heActRelu(out[i*F+f]);
    }
}

void heAvgPooling(CVec& in, int P=2) {
}

/**
 *  Dense:
 *  Y_i = X_j*W_ji * + B_i
 */
void heDense(CVec& out, const CVec& in, Layer& l, int s=1) {
    const int J=l.wd[0];
    const int I=l.wd[1];
    cout << "Dense: " << J << " " << I << endl;
    #pragma omp parallel for
    for (int i=0; i<I; i++) {
        out[i] = cc->EvalAdd( cc->EvalMult( in[0], l.w[i] ), l.b[i] ); // do j=0 first..
        for (int j=1; j<J; j++) {
            out[i] += cc->EvalMult( in[j], l.w[j*I+i] );
        }
        heActSigmoid(out[i]);
    }
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

    int ncore = omp_get_max_threads();
    cout << "===== IDASH test with cores = " << ncore << endl;
         
    usint nbp = 29000;    // # of neuclotides
    usint batchSize = 2048;
    usint N = 1;

    if (argc == 2) {
        nbp       = atoi(argv[1]);
    }
    printf("Numbers BP : %d\n", nbp);
    printf("Batch      : %d\n\n", batchSize);


    //SecurityLevel securityLevel = HEStd_NotSet; //HEStd_128_classic;
    SecurityLevel securityLevel = HEStd_128_classic;

    usint nMults = 3;    // max 2-depth tower
    usint maxdepth = 2;  // max key for s^2

    usint scaleFactor = 29; // will also effect ring dimention
    usint numLargeDigits = 0;
    usint firstModSize = 49;
    usint relinWindow = 0;   /* 0 means using CRT */
    int ringDimension = 0;   // default 8192 for std_128, but also depend on sf

    cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, scaleFactor, batchSize, securityLevel, ringDimension,
            APPROXAUTO, HYBRID, //(Rd=16384),
            //APPROXRESCALE, BV(Rd=8192),
            numLargeDigits, maxdepth, firstModSize, relinWindow, OPTIMIZED);

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);   // Not for BFV

    ccInfo(cc);
    
    int nbatch = N/batchSize + int(N%batchSize!=0);
    cout << "len(x) / batchSize / # batch : " << N << " // " << batchSize << " = " << nbatch << endl;

    TIC(t1);
    cout << "Generate keys..." << endl;
    LPKeyPair<DCRTPoly> keyPair = cc->KeyGen();
    DURATION tKG = TOC(t1);

/*
    TIC(t1);
    cout << "Generate sum key..." << endl;
    cc->EvalSumKeyGen(keyPair.secretKey);
    DURATION tSumKG = TOC(t1);
*/
    TIC(t1);
    cout << "Generate multi keys..." << endl;
    cc->EvalMultKeyGen(keyPair.secretKey);
    DURATION tMultKG = TOC(t1);

/*
    TIC(t1);
    vector<int> ilist(nbatch-1);
    #pragma omp parallel for
    for (int i=0;i<nbatch-1;i++) ilist[i]=-(i+1);
    cc->EvalAtIndexKeyGen(keyPair.secretKey, ilist);
    DURATION tRotKG = TOC(t1);
*/

    //
    // read model parameter
    //
    H5File h5d(H5NAME, H5F_ACC_RDONLY);
    Layer conv1 (h5d, "/conv1/conv1/kernel:0",   "/conv1/conv1/bias:0");
    Layer conv2 (h5d, "/conv2/conv2/kernel:0",   "/conv2/conv2/bias:0");
    Layer conv3 (h5d, "/conv3/conv3/kernel:0",   "/conv3/conv3/bias:0");
    Layer output(h5d, "/output/output/kernel:0", "/output/output/bias:0");

    int in1 = nbp;
    int CH0 = conv1.wd[1];
    int in2 = conv1.getOutDim(in1, 2, 2);
    int in3 = conv2.getOutDim(in2, 1, 2);
    int in4 = conv3.getOutDim(in3, 1, 2);
    output.setInDim(output.wd[0]);
    cout << "Output dimention...." << endl;
    cout << in1 << endl;
    cout << in2 << endl;
    cout << in3 << endl;
    cout << in4 << endl;

    // prepare ciphertext space
    CVec ctx0(conv1.wd[1]*in1); // input
    CVec ctx1(conv2.wd[1]*in2);
    CVec ctx2(conv3.wd[1]*in3);
    CVec ctx3(output.wd[0]);    // output of flatten
    CVec ctx4(output.wd[1]);    // output

    cout << "Read/Generate data and models.... " << endl;
    TIC(t);
    vector<double> x(batchSize);
    #pragma omp parallel for
    for (int i=0; i<batchSize; i++) x[i] = 1.0*(i+1); //(double) (rand()-RAND_MAX/2) / (RAND_MAX);

    #pragma omp parallel for
    for (int i=0; i<nbp*CH0; i++) {
      Plaintext ptx = cc->MakeCKKSPackedPlaintext(x);
      ctx0[i] = cc->Encrypt(keyPair.publicKey, ptx);
    }
    
    DURATION tEncAll = TOC(t);

    cout << "Eval..." << endl;
    TIC(t);

    //ctx1[0] = cc->EvalMult(0.5, ctx0[1]);// + double(conv1.b[3]);
    heConv1d(ctx1, ctx0, conv1, 2);
    cout << "C1..." << endl;
    heConv1d(ctx2, ctx1, conv2);
    cout << "C2..." << endl;
    heConv1d(ctx3, ctx2, conv3);
    cout << "C3..." << endl;
    heDense(ctx4, ctx3, output);
    cout << "D1..." << endl;

    DURATION tEvalAll = TOC(t);
    cout << endl;

    cout << "Decrypt... ";
    TIC(t);
    Plaintext res;
    cc->Decrypt(keyPair.secretKey, ctx4[0], &res);
    cout << res;
    DURATION tDec = TOC(t);


    //double he = res->GetCKKSPackedValue()[0].real();
  

    cout << "=== Timing (s):" << endl;
    //cout << "RotKey  Gen        : " << tRotKG.count() <<endl;
    //cout << "SumKey  Gen  : " << tSumKG.count() <<endl;
    cout << "Pub/Sec Gen  : " << tKG.count() <<endl;
    cout << "MultKey Gen  : " << tMultKG.count() <<endl;
    cout << "Pack/Enc /np : " << tEncAll.count()/nbp <<endl;
    cout << "Eval     /np : " << tEvalAll.count()/nbp <<endl;
    cout << "Decrypt      : " << tDec.count() <<endl;

    printMemoryUsage();
    return 0;
}
