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
DURATION t_input, t_enc, t_dec, t_square, t_batch_sum, t_reduct_sum;   /// computation related.
DURATION t_raw, t_total;

using namespace std;
using namespace lbcrypto;

int main(int argc, char* argv[]) {

    usint batchSize = 8192;
    usint scaleFactor = 39;
    int ringDimension = 2* batchSize;

    TIC(t_global);
    
    //**************************************************
    // Instantiate the crypto context and key generation
    //**************************************************
    SecurityLevel securityLevel = HEStd_128_classic;
    //SecurityLevel securityLevel = HEStd_NotSet;

#define FASTROT
#ifdef FASTROT
    usint nMults   = 1;        // max depth of tower
#else
    usint nMults   = 2;        // max depth of tower
#endif
    usint maxdepth = 1;        // max key for s^i 
    /// maxdepth : determines the capability of relinearization key.

    usint numLargeDigits = 0;
    int relinwin = 0;

    cout << "Initializing crypto context..." << endl;

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
    cout << "   Ring dimension : " << cc->GetRingDimension() << endl;
    cout << "   p     : " << ccParam->GetPlaintextModulus() << endl;
    cout << "   log2q : " << log2(cc->GetModulus().ConvertToDouble())  << endl;
    
    cout << "=================================================" << endl;
    cout << "Plaintext  Modulus : " << ccParam -> GetPlaintextModulus() << endl;
//     cout << "Ciphertext Modulus : " << ccParam -> GetCiphertextModulus() << endl; /// doesn't work? 
    cout << "Cyclotomic Order   : " << ccParam -> GetElementParams() -> GetCyclotomicOrder() << endl;
    cout << "   log2q : " << log2( cc-> GetModulus().ConvertToDouble()) << endl;
    cout << "   ScaleFactor    : " << scaleFactor << endl;
    cout << "   ScaleFactor[T] : " << cc-> GetEncodingParams() << endl;
    cout << "=================================================" << endl;
    
    
    /// Load the binary file as input.
    //TIC(t_tmp);
    //vector<double> x = read_bin(input_bin);
    //unsigned long N = x.size();
    //t_input = TOC(t_tmp);
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
#ifndef FASTROT
    cc->EvalSumKeyGen(keyPair.secretKey);
#endif
    DURATION tSumKG = TOC(t1);

    TIC(t1);
    cout << "Generating mult key ..." << endl;
    cc->EvalMultKeyGen(keyPair.secretKey);
    DURATION tMultKG = TOC(t1);

    //********************************************************
    //
    // Full rotation key for merging partial sum in each batch
    //
    //********************************************************
    TIC(t1);
    cout << "Generating full rotation keys ..." << endl << endl;
#ifdef FASTROT
    int nr = log2(batchSize);
    vector<int> ilist(nr);
    for (int i=0;i<nr;i++)   ilist[i] = 1<<i;
    cc->EvalAtIndexKeyGen(keyPair.secretKey, ilist);
#else
    vector<int> ilist(num_batch-1);
    #pragma omp parallel for
    for (int i=0;i<num_batch-1;i++) ilist[i]=-(i+1);
    cc->EvalAtIndexKeyGen(keyPair.secretKey, ilist);
#endif
    DURATION tRotKG = TOC(t1);

    //**************************************************
    //
    // Caculate \sum sigmoid(x) w/o HE
    //
    //**************************************************
    cout << "Native computation w/o HE ..." << endl;
    double exact_sum = 0.;
    
    TIC(t);
    #pragma omp parallel for reduction(+ : exact_sum)
    for (usint i=0; i<N; i++) {
#ifdef SQUARE
        exact_sum += x[i]*x[i];
#else
        exact_sum += x[i];
#endif
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
    
    #pragma omp parallel for
    for (int i = 0; i < num_batch; i++) {
#ifdef SQUARE
#if defined(FASTROT)
        //bctsum[i] = cc->EvalMult(ctx[i], ctx[i]);
        bctsum[i] = cc->EvalMultAndRelinearize(ctx[i], ctx[i]);
        Ciphertext<DCRTPoly> crot;
	for (int j=1;j<=nr;j++) {
    	    crot = cc->EvalAtIndex(bctsum[i], (1<<(nr-j)));
	    cc->EvalAddInPlace(bctsum[i], crot);
	}                
#else
        auto x2   = cc->EvalMult(ctx[i], ctx[i]);
        bctsum[i] = cc->EvalSum(x2, batchSize);
#endif

#else

#if defined(FASTROT)
        bctsum[i] = ctx[i];
	//auto cPrecomp = cc->EvalFastRotationPrecompute(bctsum[i]);
        Ciphertext<DCRTPoly> crot;
	for (int j=1;j<=nr;j++) {
    	    crot = cc->EvalAtIndex(bctsum[i], (1<<(nr-j)));
    	    //crot = cc->EvalFastRotation(bctsum[i], 1<<(nr-j), M, cPrecomp);
	    cc->EvalAddInPlace(bctsum[i], crot);
	}
#else
        bctsum[i] = cc->EvalSum(ctx[i], batchSize);
#endif
#endif
    }
    cout <<endl << endl;
    DURATION tEvalAll = TOC(t);
    TIC(t);
    cout << "Merging partial sum." << endl << endl;
#ifdef FASTROT
    auto ctsum = bctsum[0];
    for (int i=1;i<num_batch; i++) {
        cc->EvalAddInPlace(ctsum, bctsum[i]);
    }
#else
    auto ctsum = cc -> EvalSum(cc->EvalMerge(bctsum), num_batch);
#endif    
    
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
    
    /// align palisade & seal.
    t_pk  = tKG;
    t_gk  = tRotKG;
    t_rk  = tMultKG;
    t_enc = tEncAll;
    t_dec = tDec;
    t_square = tEvalAll;
    t_batch_sum  = tEvalAll;
    t_reduct_sum = tMerge;
    double raw_result = exact_sum;
    
    double error = he_result - raw_result;
        
                printf( "CKKS Result : %.6f\n", he_result);
                    printf( "Raw  Result : %.6f\n", raw_result);
        
    vector<double> out_vec(20);
    
    /// statistics for key generation.
    out_vec[0] = t_pk.count();
    out_vec[1] = t_sk.count();
    out_vec[2] = t_gk.count();
    out_vec[3] = t_rk.count();  
    \\
    /// statistics for computations.
    out_vec[4] = t_input.count();
    out_vec[5] = t_enc.count();
    out_vec[6] = t_dec.count();
    out_vec[7] = t_square.count(); //t_square.count(); 
    out_vec[8] = t_batch_sum.count(); //t_batch_sum.count(); 
    out_vec[9] = t_reduct_sum.count(); //t_reduct_sum.count();
    out_vec[10] = t_raw.count();
    out_vec[11] = t_total.count();
    out_vec[12] = raw_result;
    out_vec[13] = he_result;
    out_vec[14] = num_batch;
    out_vec[15] = tSumKG.count();
    
    
    cout << "======== Performance    ========" << endl;
    cout << "======== Key Generation ========" << endl;
    cout << "Public Key        : " << t_pk.count() <<endl;
    cout << "Secret Key Gen    : " << t_sk.count() <<endl;
    cout << "Rotation Key      : " << t_gk.count() <<endl;
    cout << "Relinearilzation  : " << t_rk.count() <<endl;
    
    cout << "======== Computations   ========" << endl;
    cout << "Prepare Input     : " << t_input.count() <<endl;
    cout << "Encode/Encrypt    : " << t_enc.count() <<endl;
    cout << "Decode/Decrypt    : " << t_dec.count() <<endl;
    cout << "Enc. Squaring     : " << t_square.count() <<endl;
    cout << "Enc. Sum [batch]  : " << t_batch_sum.count() / num_batch <<endl;
    cout << "Enc. Sum [reduce] : " << t_reduct_sum.count() <<endl;
    cout << "Error             : " << setprecision(16) << error << endl;
    cout << "# of Cores        : " << omp_get_max_threads() << endl;
    cout << tSumKG.count();
    
    
    return 0;
}
