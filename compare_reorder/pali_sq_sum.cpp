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

    usint num_batch = 1;
    usint nMults   = 1;        // max depth of tower
    usint maxdepth = 2;        // max key for s^i : determines the capability of relinearization key.
    usint scaleFactor = 39;
    usint batchSize = 8192;
    int FIRSTBIT = 60;
    usint numLargeDigits = nMults + 1;
    int relinwin = 10;

    if (argc > 1) num_batch   = atoi(argv[1]);
    if (argc > 2) nMults      = atoi(argv[2]);
    if (argc > 3) batchSize   = atoi(argv[3]);
    if (argc > 4) scaleFactor = atoi(argv[4]);
    if (argc > 5) FIRSTBIT    = atoi(argv[5]);
    if (argc > 6) numLargeDigits = atoi(argv[6]);
    if (argc > 7) relinwin = atoi(argv[7]);

    srand(time(0));
    int ringDimension = 16384;//2* batchSize;

    TIC(t_global);
    
    SecurityLevel securityLevel = HEStd_128_classic;
    //SecurityLevel securityLevel = HEStd_NotSet;


    auto cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, scaleFactor, batchSize, securityLevel, ringDimension,
            APPROXRESCALE, BV, numLargeDigits, maxdepth, FIRSTBIT, relinwin, OPTIMIZED);
            //APPROXAUTO APPROXRESCALE EXACTRESCALE,       BV(Rd=8192),HYBRID

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);

    // Output the generated parameters
    auto ccParam = cc->GetCryptoParameters();
    cout << "# of Cores        : " << omp_get_max_threads() << endl;
    cout << "   Ring dimension : " << cc->GetRingDimension() << endl;
    cout << "   log2q : " << log2( cc-> GetModulus().ConvertToDouble()) << endl ; //; + FIRSTBIT << endl;
    cout << "  log2 q : " << ccParam->GetElementParams()->GetModulus().GetMSB() << endl;
                                                              
    cout << "   ScaleFactor[T] : " << cc->GetEncodingParams() << endl;
    
    int N = batchSize*num_batch;
    
    std::vector<double> x(N);
    for (int i=0; i<N; i++) {
       x[i] = double(rand())/RAND_MAX - 0.5;
       //x[i] = double(1.0);
     }
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
    for (usint i=0; i<N; i++) raw_result += x[i]*x[i];
    DURATION t_raw = TOC(t);

    //**************************************************
    //
    // Packing data into ciphertexts
    //
    //**************************************************
    TIC(t);
    vector<Ciphertext<DCRTPoly>> ctx(num_batch);

    #pragma omp parallel for
    for (int i=0; i<num_batch; i++) {
        //vector<double> xbatch = {x.begin() + i*batchSize, (i==(num_batch-1))? x.end(): x.begin() + (i+1)*batchSize };
        vector<double> xbatch = {x.begin() + i*batchSize, x.begin() + (i+1)*batchSize };
        ctx[i] = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(xbatch) );
    }
    DURATION tEncAll = TOC(t);
    cout << endl << endl;

    //**************************************************
    //
    // HE partial sum per batch
    //
    //**************************************************
    TIC(t);
    Ciphertext<DCRTPoly>  ctx2 = cc->EvalMultNoRelin(ctx[0], ctx[0]);
    //cc->RescaleInPlace(ctx2);

    for (int i = 1; i < num_batch; i++) {
        Ciphertext<DCRTPoly> tmp_ctx2 = cc->EvalMultNoRelin(ctx[i], ctx[i]);
        cc->EvalAddInPlace(ctx2, tmp_ctx2);
    }

    DURATION tEvalAll = TOC(t);

    TIC(t);
    ctx2 = cc->Relinearize(ctx2);  // for EvalSum
    //cc->RescaleInPlace(ctx2);
    auto ctsum = cc->EvalSum(ctx2, batchSize);
    DURATION tMerge = TOC(t);

    TIC(t);
    Plaintext res;
    cc->Decrypt(keyPair.secretKey, ctsum, &res);
    res->SetLength(1);
      
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
    double rerr  = (he_result - raw_result)/raw_result;
        
    printf( "CKKS Result : %.6f\n", he_result);
    printf( "Raw  Result : %.6f\n", raw_result);
    printf( "Rel Error (%) : %.3g\n", rerr*100);
    cout << res << endl;    // GetPlaintextModulus() - m_logError;
    

    cout << "======== Time in secs ========" << endl;
    cout << "Public Key        : " << t_pk.count() <<endl;
    cout << "Secret Key Gen    : " << t_sk.count() <<endl;
    cout << "Rotation Key      : " << t_gk.count() <<endl;
    cout << "Relinearilzation  : " << t_rk.count() <<endl<< endl;
    
    cout << "Prepare Input     : " << t_input.count() <<endl;
    cout << "Encode/Encrypt    : " << t_enc.count() <<endl;
    cout << "Decode/Decrypt    : " << t_dec.count() <<endl;
    cout << "Compute  per b    : " << t_batch_sum.count() / num_batch <<endl;
    cout << "EvalSum           : " << t_reduct_sum.count()  <<endl;

    cout << "Total: " << t_total.count()  <<endl;
    cout << "[PALISADE_Summary] " << num_batch  << " " 
                         << t_pk.count() << " "  <<  t_gk.count()<< " " << t_rk.count() << " "
                         << t_enc.count() << " " << t_dec.count()<< " " << t_batch_sum.count()<< " " <<  t_reduct_sum.count() << " " 
                         << t_total.count() << " " << t_raw.count() << " " << error << " "
                         << batchSize << " "<<  scaleFactor << " " << FIRSTBIT << " " << rerr << endl;

    return 0;
}
