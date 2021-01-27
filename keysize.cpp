#include <palisade/pke/palisade.h>
#include <palisade/pke/ciphertext-ser.h>
#include <palisade/pke/cryptocontext-ser.h>
#include <palisade/pke/pubkeylp-ser.h>
#include <palisade/pke/scheme/ckks/ckks-ser.h>
#include <palisade/core/utils/serialize-binary.h>

#include <iostream>
#include <ctime>
#include <cstdio>
#include <sstream>

using std::cout;
using std::endl;
using std::string;

using namespace lbcrypto;

void benchkeysize(int sf, int bs) {
    cout << "============ Process .. sf: " << sf << "  bs: " << bs  <<endl;
    int param = bs;

    char fn[20];

    int batchSize = bs;
    int N= batchSize/2;
    double a0 = 0.1;

    // Instantiate the crypto context
    SecurityLevel securityLevel = HEStd_128_classic;
    double sigma = 3.2;
    int maxdepth = 2;

    int plaintextModulus = 65537; //   (q-1)/ cyclom \in N
    //int plaintextModulus = 131073;  //   (q-1)/m = N, m=16384 (32768)
    //usint plaintextModulus = 536903681;
    int nMults = 2;
    int ringD = 0;
    int numLargeDigits = 0;
    int dcrtBits = 0; ///0
    int relinWindow = 0;
    int firstModSize = 60; 

    // BGV/BFS does not support serialization !!??
    //#define BGV
#if defined(BFV)
    CryptoContext<DCRTPoly> cc = CryptoContextFactory<DCRTPoly>::genCryptoContextBFVrnsB(
            plaintextModulus, securityLevel, sigma, 0, nMults, 0, OPTIMIZED, maxdepth);
#elif defined(BGV)
    CryptoContext<DCRTPoly> cc = CryptoContextFactory<DCRTPoly>::genCryptoContextBGVrns(
            nMults, plaintextModulus, securityLevel, sigma, maxdepth, OPTIMIZED, HYBRID,
            ringD, numLargeDigits, firstModSize, dcrtBits, relinWindow); //, batchSize, AUTO);
#else
    int scalef = sf;
    CryptoContext<DCRTPoly> cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, scalef, batchSize, securityLevel, ringD,
            /*EXACTRESCALE*/ APPROXRESCALE, HYBRID, 
            numLargeDigits, maxdepth, firstModSize, relinWindow, OPTIMIZED);
#endif

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
#if !defined(BFV)
    cc->Enable(LEVELEDSHE);
#endif

    // Output the generated parameters
    auto ccParam = cc->GetCryptoParameters();
    std::cout << "ring dimension : " << cc->GetRingDimension() << std::endl;
    std::cout << "cyclo order    : " << cc->GetCyclotomicOrder() << std::endl;
    //std::cout << "GetRootOfUnity : " << cc->GetRootOfUnity() << std::endl;
    std::cout << "p, log2 q = " << ccParam->GetPlaintextModulus() << " " << log2(cc->GetModulus().ConvertToDouble())  << std::endl;
    std::cout << "Size of ptx: " << N*sizeof(a0)/1024.0 << " kB" << std::endl;

    // Step 2 : Initialize Public Key Containers & Generate a public/private key pair
    LPKeyPair<DCRTPoly> kp = cc->KeyGen();
    cc->EvalSumKeyGen(kp.secretKey);
    cc->EvalMultKeyGen(kp.secretKey);

    vector<int> ilist(bs-1);
    for (int i=0;i<bs-1;i++) ilist[i]=-(i+1);
    cc->EvalAtIndexKeyGen(kp.secretKey, ilist);

    //=========================================
#if defined(BGV)|| defined(BFV)
    std::vector<int64_t> x(N, a0);
    for (int i=0; i<N; i++) x[i] = int(a0);
    Plaintext ptx = cc->MakePackedPlaintext(x);
#else
    std::vector<double> x(N, a0);
    for (int i=0; i<N; i++) x[i] = a0;
    Plaintext ptx = cc->MakeCKKSPackedPlaintext(x);
#endif
    Ciphertext<DCRTPoly> ctx = cc->Encrypt(kp.publicKey, ptx);

    //std::vector<string> lable = {"Context","Cipher", "PubKey", "SecKey", "SumKey", "RotKey","MultKey"};
    std::vector<double> field;

    {
        std::stringstream ss;
        Serial::Serialize(cc, ss, SerType::BINARY);
        field.push_back(ss.str().size());
    }
    {
        std::stringstream ss;
        Serial::Serialize(ctx, ss, SerType::BINARY);
        field.push_back(ss.str().size());
    }
    {
        std::stringstream ss;
        Serial::Serialize(kp.publicKey, ss, SerType::BINARY);
        field.push_back(ss.str().size());
    }
    {
        std::stringstream ss;
        Serial::Serialize(kp.secretKey, ss, SerType::BINARY);
        field.push_back(ss.str().size());
    }
    {
        std::stringstream ss;
        cc->SerializeEvalSumKey(ss, SerType::BINARY);
        field.push_back(ss.str().size());
    }
    {
        std::stringstream ss;
        cc->SerializeEvalAutomorphismKey(ss, SerType::BINARY);
        field.push_back(ss.str().size());
    }
    {
        std::stringstream ss;
        cc->SerializeEvalMultKey(ss, SerType::BINARY);
        field.push_back(ss.str().size());
    }

    cc->ClearEvalSumKeys();
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();

    cout << "Ctx size   : " << field[1] << endl;
    cout << "Pub/Sec key: " << field[2] << " " << field[3] << endl;

    cout << "[SUMMARY]\t" << bs << "\t" << sf << "\t"
        << cc->GetRingDimension() << "\t"
        << cc->GetCyclotomicOrder() << "\t"
        << cc->GetRootOfUnity() << "\t"
        << ccParam->GetPlaintextModulus() << "\t"
        << log2(cc->GetModulus().ConvertToDouble()) << "\t";

    for(int i : field) cout << i << "\t";
    cout << endl;


}

int main(int argc, char* argv[]) {

    int maxp = 4;
    int sf = 39;
    if (argc == 3) {
        sf   = atoi(argv[1]);
        maxp = atoi(argv[2]);
    }

    for (int i=0; i<=maxp; i++) benchkeysize(sf, 1<<i);



    return 0;
}


