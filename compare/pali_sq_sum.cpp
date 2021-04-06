#include <palisade/pke/palisade.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <chrono>
#include <omp.h>
typedef  std::chrono::duration<double> DURATION;
std::chrono::time_point<std::chrono::system_clock>  tc, tw, t, t1;
#undef TIC
#undef TOC
#define TIC(t) t = std::chrono::system_clock::now();
#define TOC(t) (std::chrono::system_clock::now() - t)

std::chrono::time_point<std::chrono::system_clock> t_tmp, t_global;

DURATION t_pk, t_sk, t_rk, t_gk;        /// key related.
DURATION t_input, t_enc, t_dec, t_batch_sum, t_reduct_sum;   /// computation related.
DURATION t_raw, t_total;

using namespace std;
using namespace lbcrypto;

int main(int argc, char* argv[]) {

    usint batchSize = 8192;
    usint scaleFactor = 39;
    int ringDimension = 2* batchSize;

    TIC(t_global);
    
    SecurityLevel securityLevel = HEStd_128_classic;
    //SecurityLevel securityLevel = HEStd_NotSet;

    usint nMults   = 1;        // max depth of tower
    usint maxdepth = 1;        // max key for s^i : determines the capability of relinearization key.
    usint numLargeDigits = nMults + 1;
    int relinwin = 0;

    auto cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, scaleFactor, batchSize, securityLevel, ringDimension,
            APPROXAUTO, HYBRID,
            numLargeDigits, maxdepth, 60, relinwin, OPTIMIZED);
            //APPROXAUTO APPROXRESCALE EXACTRESCALE,       BV(Rd=8192),HYBRID

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);

    // Output the generated parameters
    auto ccParam = cc->GetCryptoParameters();
    cout << "# of Cores        : " << omp_get_max_threads() << endl;
    cout << "   Ring dimension : " << cc->GetRingDimension() << endl;
    cout << "   log2q : " << log2( cc-> GetModulus().ConvertToDouble()) << endl;
    cout << "   ScaleFactor[T] : " << cc->GetEncodingParams() << endl;
    
    
    int num_batch = 10;
    int N = batchSize*num_batch;
    
    std::vector<double> x(N);
    #pragma omp parallel for
    for (int i=0; i<N; i++) x[i] = (double) (rand()-RAND_MAX/2) / (RAND_MAX);

    cout << "   len(x) / batchSize / #batch : " << N << " // " << batchSize << " = " << num_batch << endl << endl;

    //**************************************************
    //
    // Key generation
    //
    //**************************************************
    TIC(t1);
    cout << "Generating key pair ..." << endl;
    LPKeyPair<DCRTPoly> keyPair = cc->KeyGen();
    DURATION tKG = TOC(t1);

    TIC(t1);
    cout << "Generating sum key ..." << endl;
    cc->EvalSumKeyGen(keyPair.secretKey);
    DURATION tSumKG = TOC(t1);

    TIC(t1);
    cout << "Generating mult key ..." << endl;
    cc->EvalMultKeyGen(keyPair.secretKey);
    DURATION tMultKG = TOC(t1);


    //**************************************************
    //
    // Caculate \sum x^2 w/o HE
    //
    //**************************************************
    cout << "Native computation w/o HE ..." << endl;
    double raw_result = 0.;
    
    TIC(t);
    #pragma omp parallel for reduction(+ : raw_result)
    for (usint i=0; i<N; i++) {
        raw_result += x[i]*x[i];
    }
    DURATION t_raw = TOC(t);

    //**************************************************
    //
    // Packing data into ciphertexts
    //
    //**************************************************
    TIC(t);
    cout << "Pack and encrypt data by batch ";
    vector<Ciphertext<DCRTPoly>> ctx(num_batch);

    #pragma omp parallel for
    for (int i=0; i<num_batch; i++) {
//         cout << ".";
        vector<double> xbatch = {x.begin() + i*batchSize, (i==(num_batch-1))? x.end(): x.begin() + (i+1)*batchSize };
        Plaintext ptx = cc->MakeCKKSPackedPlaintext(xbatch);
        ctx[i] = cc->Encrypt(keyPair.publicKey, ptx);
    }
    DURATION tEncAll = TOC(t);
    cout << endl << endl;

    //**************************************************
    //
    // HE partial sum per batch
    //
    //**************************************************
    TIC(t);
    cout << "Compute batch sum ";
    vector<Ciphertext<DCRTPoly>> bctsum(num_batch);
    //int num_rotation = log2(batchSize);
    //int M = 2*cc->GetRingDimension();
    DURATION t_square;
    #pragma omp parallel for
    for (int i = 0; i < num_batch; i++) {

    TIC(t1);
        auto ctx2 = cc->EvalMult(ctx[i], ctx[i]);
    t_square += TOC(t1);
    bctsum[i] = cc->EvalSum(ctx2, batchSize);

    }
    cout <<endl << endl;
    DURATION tEvalAll = TOC(t);

    // TODO: OMP optimization
    TIC(t);
    cout << "Merging partial sum." << endl << endl;
    auto ctsum = bctsum[0];
    for (int i=1;i<num_batch; i++) {
        cc->EvalAddInPlace(ctsum, bctsum[i]);
    }
    DURATION tMerge = TOC(t);


    TIC(t);
    Plaintext res;
    cc->Decrypt(keyPair.secretKey, ctsum, &res);
    DURATION tDec = TOC(t);
    double he_result = res -> GetCKKSPackedValue()[0].real();
    
    t_total = TOC(t_global);
    
    
    /************************************
        Summarize the results.
    *************************************/
    
    t_pk  = tKG;
    t_gk  = tSumKG;
    t_rk  = tMultKG;
    t_enc = tEncAll;
    t_dec = tDec;
    t_batch_sum  = tEvalAll;
    t_reduct_sum = tMerge;
    
    double error = he_result - raw_result;
        
    printf( "CKKS Result : %.6f\n", he_result);
    printf( "Raw  Result : %.6f\n", raw_result);
        
    cout << "======== Key Generation ========" << endl;
    cout << "Public Key        : " << t_pk.count() <<endl;
    cout << "Secret Key Gen    : " << t_sk.count() <<endl;
    cout << "Rotation Key      : " << t_gk.count() <<endl;
    cout << "Relinearilzation  : " << t_rk.count() <<endl;
    
    cout << "======== Computations   ========" << endl;
    cout << "Prepare Input     : " << t_input.count() <<endl;
    cout << "Encode/Encrypt    : " << t_enc.count() <<endl;
    cout << "Decode/Decrypt    : " << t_dec.count() <<endl;
    cout << "Squaring per b    : " << t_square.count() / num_batch <<endl;
    cout << "Compute  per b    : " << t_batch_sum.count() / num_batch <<endl;
    cout << "Enc. Sum [reduce] : " << t_reduct_sum.count() <<endl;
    cout << "Error             : " << setprecision(16) << error << endl;
    
    
    return 0;
}
