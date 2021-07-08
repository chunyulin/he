#include <palisade/pke/palisade.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
using std::cout;
using std::endl;
using std::vector;
using std::setw;
using std::setprecision;

#include <chrono>
#include <omp.h>
typedef  std::chrono::duration<double> DURATION;
std::chrono::time_point<std::chrono::system_clock>  tc, tw, t, t1;

#undef TIC
#undef TOC
#define TIC(t) t = std::chrono::system_clock::now();
#define TOC(t) (std::chrono::system_clock::now() - t)

using namespace lbcrypto;


int main(int argc, char* argv[]) {

    int ncore = omp_get_max_threads();
    cout << "# of Cores        : " << ncore << endl;
       
    usint batchSize = 8192;
    usint N = batchSize*4;

    if (argc == 3) {
        batchSize = atoi(argv[1]);
        N         = int(batchSize * atof(argv[2]));
    }

    double a0 = 1.0;
    //usint N = 16384;

    // Instantiate the crypto context
    usint plaintextModulus = 536903681;
    //SecurityLevel securityLevel = HEStd_NotSet; //HEStd_128_classic;
    SecurityLevel securityLevel = HEStd_128_classic;
    double sigma = 3.2;

    usint nMults = 1;    // max 2-depth tower
    usint maxdepth = 1;  // max key for s^2

    usint numLargeDigits = 0;
    usint firstModSize = 0;
    usint relinWindow = 0;   /* 0 means using CRT */
    int ringDimension = 0;   // default 8192 for std_128, but also depend on sf
    usint dcrtBits = 60;
    //usint plaintextModulus = 32768*32+1;

#define BFV
#if defined(BFV)
    CryptoContext<DCRTPoly> cc = CryptoContextFactory<DCRTPoly>::genCryptoContextBFVrnsB(
            plaintextModulus, securityLevel, sigma, 0, nMults, 0, OPTIMIZED, maxdepth,
            relinWindow, dcrtBits, ringDimension);
#elif defined(BGV)
    CryptoContext<DCRTPoly> cc = CryptoContextFactory<DCRTPoly>::genCryptoContextBGVrns(
            nMults, plaintextModulus, securityLevel, sigma, maxdepth, OPTIMIZED, HYBRID,
            ringDimension, numLargeDigits, firstModSize, dcrtBits, relinWindow, batchSize, AUTO);
    /* ModReduceInternal has not been enabled for this LEV FHE  scheme. */
#else
    usint scaleFactor = 49; // will also effect ring dimention
    auto cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, scaleFactor, batchSize, securityLevel, ringDimension,
            APPROXRESCALE, BV,  //(Rd=16384),
            //APPROXRESCALE, BV(Rd=8192),
            numLargeDigits, maxdepth, 60, 5, OPTIMIZED);
#endif

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
#if !defined(BFV)
    cc->Enable(LEVELEDSHE);   // Not for BFV
#endif

    // Output the generated parameters
    auto ccParam = cc->GetCryptoParameters();
    std::cout << "======= Simple HE test ==============================================" << endl;
    std::cout << "Ring dimension : " << cc->GetRingDimension() << std::endl;
    //std::cout << "cyclo order    : " << cc->GetCyclotomicOrder() << std::endl;
    //std::cout << "GetRootOfUnity : " << cc->GetRootOfUnity() << std::endl;
    std::cout << "p, log2 q = " << ccParam->GetPlaintextModulus() << " " << log2(cc->GetModulus().ConvertToDouble())  << std::endl;

    int rlen = N/batchSize + int(N%batchSize!=0);
    cout << "len(x) / batchSize / # batch : " << N << " // " << batchSize << " = " << rlen << endl;

{
    std::vector<int64_t> tmp = {0,1,2,3};
    Plaintext ptx = cc->MakePackedPlaintext(tmp);
    cout << "Log/High = " << ptx->LowBound() << " " << ptx->HighBound() << endl;
    cout << "GetElementModulus() = " << ptx->GetElementModulus() << endl;

}


    TIC(t1);
    LPKeyPair<DCRTPoly> keyPair = cc->KeyGen();
    DURATION tKG = TOC(t1);

    TIC(t1);
    cc->EvalSumKeyGen(keyPair.secretKey);
    DURATION tSumKG = TOC(t1);

    TIC(t1);
    cc->EvalMultKeyGen(keyPair.secretKey);
    DURATION tMultKG = TOC(t1);

    TIC(t1);
    vector<int> ilist(rlen-1);
#pragma omp parallel for
    for (int i=0;i<rlen-1;i++) ilist[i]=-(i+1);
    cc->EvalAtIndexKeyGen(keyPair.secretKey, ilist);
    DURATION tRotKG = TOC(t1);


    cout << "Generating DP random vector with Sum(x^2)= ";
#if defined(BFV) || defined(BGV)
    std::vector<long> x(N);
    #pragma omp parallel for
    for (int i=0; i<N; i++) x[i] = int( (rand()-RAND_MAX/2.0) / (RAND_MAX) * 20);
    long exact_sum = 0;
#else
    std::vector<double> x(N);
    #pragma omp parallel for
    for (int i=0; i<N; i++) x[i] = (double) (rand()-RAND_MAX/2) / (RAND_MAX);
    double exact_sum = 0;
#endif

    usint  exact_flop = 2*N;
    TIC(t);
#pragma omp parallel for reduction(+ : exact_sum)
    for (int i=0; i<N; i++) {
        exact_sum += x[i]*x[i];
    }
    DURATION tDP = TOC(t);
    cout << setprecision(16) << exact_sum << endl;
#if defined(BFV) || defined(BGV)
    if (exact_sum >= ccParam->GetPlaintextModulus() ) {
        cout << "!!!!!! WARNING for small p !!! " << endl;
    }
#endif

    TIC(t);
    cout << "Packing data by batch ";
    std::vector<Ciphertext<DCRTPoly>> ctx(rlen);

#pragma omp parallel for
    for (int i=0; i<rlen; i++) {
        cout << ".";

#if defined(BFV) || defined(BGV)
        std::vector<int64_t> xbatch = {x.begin() + i*batchSize, (i==(rlen-1))? x.end(): x.begin() + (i+1)*batchSize };
        Plaintext ptx = cc->MakePackedPlaintext(xbatch);
#else
        std::vector<double> xbatch = {x.begin() + i*batchSize, (i==(rlen-1))? x.end(): x.begin() + (i+1)*batchSize };
        Plaintext ptx = cc->MakeCKKSPackedPlaintext(xbatch);
#endif
        ctx[i] = cc->Encrypt(keyPair.publicKey, ptx);
    }
    DURATION tEncAll = TOC(t);
    cout <<endl;

    cout << "Compute sum per batch "<< endl;
    TIC(t);
    Ciphertext<DCRTPoly>  ctx2 = cc->EvalMult(ctx[0], ctx[0]);
    //cc->RescaleInPlace(ctx2);
    
    for (int i = 1; i < rlen; i++) {
        Ciphertext<DCRTPoly> tmp_ctx2 = cc->EvalMult(ctx[i], ctx[i]);
        cc->EvalAddInPlace(ctx2, tmp_ctx2);
    }
    DURATION tEvalAll = TOC(t);
    
    cout << "Merge "  << endl;
    TIC(t);
    //ctx2 = cc->Relinearize(ctx2);  // for EvalSum
    //cc->RescaleInPlace(ctx2);
    auto ctsum = cc->EvalSum(ctx2, batchSize);
    DURATION tMerge = TOC(t);

#if 0
    Plaintext res11;
    for (int i=0; i<rlen; i++) {
        cc->Decrypt(keyPair.secretKey, bctsum[i], &res11);
        cout <<  res11->GetPackedValue()[0] << " ";
    }
#endif


    cout << "Decrypting result: Sum(x^2)= ";
    TIC(t);
    Plaintext res;
    cc->Decrypt(keyPair.secretKey, ctsum, &res);
    DURATION tDec = TOC(t);

#if defined(BFV) || defined(BGV)
    long he = res->GetPackedValue()[0];
#else
    double he = res->GetCKKSPackedValue()[0].real();
#endif

    std::cout << setprecision(16) << he << "  HE-EXACT= "
        << setprecision(6) << (he-exact_sum)
        << "  ( Rel. err= "<< 100.0*(he-exact_sum)/exact_sum << " % )" <<  endl;

    std::cout << "=== Timing (s):" << endl;
    std::cout << "RotKeyGen per batch: " << tRotKG.count() <<endl;
    std::cout << "SumKey  Gen        : " << tSumKG.count() <<endl;
    std::cout << "MultKey Gen        : " << tMultKG.count() <<endl;
    std::cout << "Pub/Sec Gen        : " << tKG.count() <<endl;
    std::cout << "Pack/Enc  per batch: " << tEncAll.count()/rlen <<endl;
    std::cout << "Eval      per batch: " << tEvalAll.count()/rlen <<endl;
    std::cout << "Merge     per batch: " << tMerge.count()/rlen <<endl;
    std::cout << "Dec time           : " << tDec.count() <<endl;
    printf("W/O HE             : %.4f     HE: %.2f KF  CPU: %.2f MF (%.1fx)  ) \n", tDP.count(),
                 exact_flop / (tEvalAll+tMerge).count() / ncore * 1e-3,
                 exact_flop / tDP.count() / ncore * 1e-6,
                 (tEvalAll+tMerge).count()/tDP.count() );
            



    std::cout << "[TimeSummary] " << ncore << " " 
        << cc->GetRingDimension() << " "
        << ccParam->GetPlaintextModulus() << " " << log2(cc->GetModulus().ConvertToDouble())  << " "
        << N << " " << batchSize << " " << rlen << " "
        << exact_flop << " "
        << 100.0*(he-exact_sum)/exact_sum << " "
        << tRotKG.count() << " "
        << tSumKG.count() << " "
        << tMultKG.count() << " "
        << tKG.count() << " "
        << tEncAll.count() << " "
        << tEvalAll.count() << " "
        << tMerge.count() << " " 
        << tDec.count() << " "
        << tDP.count() << endl;

    cout << endl;
    printf("doubel MFlops/core: %.4f\n", exact_flop/tDP.count()/ncore*1e-6 );
    printf("HE     KFlops/core: %.4f\n", exact_flop/(tEvalAll.count()+tMerge.count())/ncore*1e-3   );
            

    return 0;
}
