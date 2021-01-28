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

    usint ncore = omp_get_max_threads();
    usint batchSize = 8192;
    usint N = batchSize*4;
    usint scaleFactor = 39;  // will also effect ring dimention

    if (argc >= 3) {
        batchSize = atoi(argv[1]);
        N         = int(batchSize * atof(argv[2]));
    }
    if (argc == 4) {
        scaleFactor = atoi(argv[3]);
    }

    //
    // Instantiate the crypto context and key generation
    //
    SecurityLevel securityLevel = HEStd_128_classic;
    usint plaintextModulus = 536903681;
    double sigma = 3.2;

    usint nMults = 4;        // max 2-depth tower
    usint maxdepth = 3;      // max key for s^2

    usint numLargeDigits = 0;
    usint firstModSize = 0;
    usint relinWindow = 0;   /* 0 means using CRT */
    usint ringDimension = 0;   // default 8192 for std_128, but also depend on sf
    usint dcrtBits = 60;

    std::cout << "======= Simple HE sum of sigmoid(x) ======" << endl;
    std::cout << "Initializing crypto context..." << endl;

    auto cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, scaleFactor, batchSize, securityLevel, ringDimension,
            EXACTRESCALE, HYBRID,
            //APPROXRESCALE, BV(Rd=8192),
            numLargeDigits, maxdepth, 60, 5, OPTIMIZED);

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);   // Not for BFV

    // Output the generated parameters
    auto ccParam = cc->GetCryptoParameters();
    std::cout << "   Ring dimension : " << cc->GetRingDimension() << std::endl;
    std::cout << "   p     : " << ccParam->GetPlaintextModulus() << std::endl;
    std::cout << "   log2q : " << log2(cc->GetModulus().ConvertToDouble())  << std::endl;

    int num_batch = N/batchSize + int(N%batchSize!=0);
    cout << "   len(x) / batchSize / #batch : " << N << " // " << batchSize << " = " << num_batch << endl;

    cout << "Generating key pair ..." << endl;
    TIC(t1);
    LPKeyPair<DCRTPoly> keyPair = cc->KeyGen();
    DURATION tKG = TOC(t1);

    cout << "Generating sum key ..." << endl;
    TIC(t1);
    cc->EvalSumKeyGen(keyPair.secretKey);
    DURATION tSumKG = TOC(t1);

    cout << "Generating mult key ..." << endl;
    TIC(t1);
    cc->EvalMultKeyGen(keyPair.secretKey);
    DURATION tMultKG = TOC(t1);

    //
    // Prepare rotation key for merging partial sum in each batch
    //
    cout << "Generating full rotation keys ..." << endl;
    TIC(t1);
    vector<int> ilist(num_batch-1);
    #pragma omp parallel for
    for (int i=0;i<num_batch-1;i++) ilist[i]=-(i+1);
    cc->EvalAtIndexKeyGen(keyPair.secretKey, ilist);
    DURATION tRotKG = TOC(t1);


    cout << "Generating DP random vector ..." << endl;
    std::vector<double> x(N);
    #pragma omp parallel for
    for (int i=0; i<N; i++) x[i] = (double) (rand()-RAND_MAX/2) / (RAND_MAX);

    //
    // Caculate \sum sigmoid(x) w/o HE
    //
    double exact_sum = 0.;
    usint  exact_flop = 7*N;
    TIC(t);
    #pragma omp parallel for reduction(+ : exact_sum)
    for (int i=0; i<N; i++) {
        exact_sum += 0.5 + 0.25*x[i]*(1.0 + 0.08333333333333*x[i]*x[i]);
    }
    DURATION tDP = TOC(t);
    cout << "Sum(sigmoid(x)) w/o HE = " << setprecision(16) << exact_sum << endl;

    //
    // Packing data into ciphertexts
    //
    TIC(t);
    cout << "Pack and encrypt data by batch ";
    std::vector<Ciphertext<DCRTPoly>> ctx(num_batch);

    #pragma omp parallel for
    for (int i=0; i<num_batch; i++) {
        cout << ".";

        std::vector<double> xbatch = {x.begin() + i*batchSize, (i==(num_batch-1))? x.end(): x.begin() + (i+1)*batchSize };
        Plaintext ptx = cc->MakeCKKSPackedPlaintext(xbatch);
        ctx[i] = cc->Encrypt(keyPair.publicKey, ptx);
    }
    DURATION tEncAll = TOC(t);
    cout <<endl;

    //
    // HE partial sum per batch
    //
    cout << "Compute batch sum ";
    TIC(t);
    std::vector<Ciphertext<DCRTPoly>> bctsum(num_batch);
    #pragma omp parallel for
    for (int i=0; i<num_batch; i++) {
        cout << ".";

	// can be implemented inside the library to reduce overhead
        auto xo4     = cc->EvalMult(0.25, ctx[i]);
        auto x2      = cc->EvalMult(ctx[i], ctx[i]);
        auto x2o12   = cc->EvalMult(0.08333333333333, x2);
        auto x2o12p1 = cc->EvalAdd(1.0, x2o12);
        auto tmp  = cc->EvalMult(xo4, x2o12p1);
        auto pp   = cc->EvalAdd(0.5, tmp);
        auto bsum = cc->EvalSum(pp, batchSize);

        bctsum[i] = bsum;
    }
    cout <<endl;
    DURATION tEvalAll = TOC(t);

    cout << "Merging partial sum." << endl;
    TIC(t);
    auto ctsum = cc->EvalSum(cc->EvalMerge(bctsum), num_batch);
    DURATION tMerge = TOC(t);


    cout << "Decrypt HE Sum(sigmoid(x)) = ";
    TIC(t);
    Plaintext res;
    cc->Decrypt(keyPair.secretKey, ctsum, &res);
    DURATION tDec = TOC(t);

    double he = res->GetCKKSPackedValue()[0].real();

    std::cout << setprecision(16) << he << "  HE-EXACT= "
        << setprecision(6) << (he-exact_sum)
        << "  ( Rel. err= "<< 100.0*(he-exact_sum)/exact_sum << " % )" <<  endl;

    cout << "=== Timing (s):" << endl;
    cout << "RotKeyGen per batch: " << tRotKG.count() <<endl;
    cout << "SumKey  Gen        : " << tSumKG.count() <<endl;
    cout << "MultKey Gen        : " << tMultKG.count() <<endl;
    cout << "Pub/Sec Gen        : " << tKG.count() <<endl;
    cout << "Pack/Enc  per batch: " << tEncAll.count()/num_batch <<endl;
    cout << "Eval      per batch: " << tEvalAll.count()/num_batch <<endl;
    cout << "Merge     per batch: " << tMerge.count()/num_batch <<endl;
    cout << "Dec time           : " << tDec.count() <<endl;
    printf("w/o HE             : %.4f  ( %.3fx slower than peak, %.1fx faster than HE )\n", tDP.count(),
            tDP.count()/(exact_flop*1.0e-9),
            (tEvalAll+tMerge).count() / tDP.count() );

    cout << endl;
    printf("doubel MFlops/core: %.4f\n", exact_flop/tDP.count()/ncore*1e-6 );
    printf("HE     KFlops/core: %.4f\n", exact_flop/(tEvalAll.count()+tMerge.count())/ncore*1e-3   );


    cout << endl;
    cout << "[TimeSummary] " << ncore << " " 
         << cc->GetRingDimension() << " "
         << ccParam->GetPlaintextModulus() << " " << log2(cc->GetModulus().ConvertToDouble())  << " "
         << N << " " << batchSize << " " << num_batch << " "
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

    return 0;
}
