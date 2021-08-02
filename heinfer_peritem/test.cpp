#include <palisade/pke/palisade.h>
#include <iostream>
#include <ctime>
#include <cstdio>
#include <sstream>
#include <complex>

using std::cout;
using std::endl;
using std::string;
using std::complex;
using std::vector;

using namespace lbcrypto;

void test() {
    using namespace std::complex_literals;
    SecurityLevel securityLevel = HEStd_NotSet;
    //SecurityLevel securityLevel = HEStd_128_classic;
    int nMults = 4;
    int maxdepth = 3;
    int relinWindow = 0;
    int sf = 39;
    int bs = 8;
    int ringD = bs*2;

    CryptoContext<DCRTPoly> cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, 49, bs, securityLevel, ringD,
            APPROXRESCALE, HYBRID, 
            0, maxdepth, sf, relinWindow, OPTIMIZED);

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);

    auto ccParam = cc->GetCryptoParameters();
    std::cout << "ring dimension : " << cc->GetRingDimension() << std::endl;
    std::cout << "cyclo order    : " << cc->GetCyclotomicOrder() << std::endl;
    std::cout << "p, log2 q = " << ccParam->GetPlaintextModulus() << " " << log2(cc->GetModulus().ConvertToDouble())  << std::endl;


    LPKeyPair<DCRTPoly> kp = cc->KeyGen();
    cc->EvalSumKeyGen(kp.secretKey);
    cc->EvalMultKeyGen(kp.secretKey);

    vector<int> ilist(bs*2-1);
    for (int i=1;i<bs*2-1;i++) ilist[i]=i;
    cc->EvalAtIndexKeyGen(kp.secretKey, ilist);

    complex<double> z(0,1.0); 

    vector<complex<double>> x(bs, z);
    //for (int i=0; i<bs; i++) x[i] = 2+2i;
    Plaintext ptx = cc->MakeCKKSPackedPlaintext(x);
    Ciphertext<DCRTPoly> ctx = cc->Encrypt(kp.publicKey, ptx);
    ctx = cc->EvalMult(ctx,ctx);
    

        Plaintext r1;
        cc->Decrypt(kp.secretKey, ctx, &r1);
        r1->SetLength(bs);
    
        auto v1 = r1->GetCKKSPackedValue();
        for (int p=0;p<bs;p++) cout << v1[p] << " ";
        cout  << endl;



    for (int i=1;i<bs*2;i++) {
    
        auto c1 = cc->EvalAtIndex(ctx, i);
    
        Plaintext res;
        cc->Decrypt(kp.secretKey, c1, &res);
        res->SetLength(bs);
        //cout << res;
    
        auto v = res->GetCKKSPackedValue();
        cout  << "=" << i << "= ";
        for (int p=0;p<bs;p++) cout << v[p] << " ";
        cout  << endl;
    }
    
    
    

}

int main(int argc, char* argv[]) {

    test();

    return 0;
}


