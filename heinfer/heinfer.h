#pragma once

#define  SANITY_CHECK

#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::setw;
using std::setprecision;


// Headers for Palisade ------------------------------------------
#include <palisade/pke/palisade.h>
using namespace lbcrypto;
typedef vector<Ciphertext<DCRTPoly>> CVec;


// Headers  for timing ------------------------------------------
#include <chrono>
typedef std::chrono::time_point<std::chrono::system_clock> TIMER;
typedef std::chrono::duration<double> DURATION;
#undef TIC
#undef TOC
#define TIC(t) t = std::chrono::system_clock::now();
#define TOC(t) (std::chrono::system_clock::now() - t)

//#define AUTORESCALE
#include <H5Cpp.h>
using namespace H5;

class Layer;

class HEInfer {

  public:

    int ninf, nbp;
    int PF;   // packing factor
    int pinf; // packing of inference, the closest power of 2 of 'ninf'

    CryptoContext<DCRTPoly> cc;
    LPKeyPair<DCRTPoly> keyPair;
    const SecurityLevel securityLevel = HEStd_128_classic;
    int nMults;    // max 2-depth tower
    int maxdepth;  // max key for s^2
    int scaleFactor; // will also effect ring dimention
    int firstModSize;
    int numLargeDigits = 0;
    usint relinWindow = 0;   /* 0 means using CRT */
    int ringDim = 0;    // delayed gived by CryptoContext

  public:

    HEInfer(int ninf_, int nbp_, int nMults, int depth, int sf, int firstmod, int ringdim, int pinf_);
    ~HEInfer() { };
    
    void test();

    void genKeys();
    void ConvReluAP1d(CVec& out, const CVec& in, Layer& l, const int Stride, const int Pool);
    void DenseSigmoid(CVec& out, const CVec& in, Layer& l);
    void DenseSigmoid_packed(CVec& out, const CVec& in, Layer& l);

    void EncodeEncrypt(CVec& cvec, const vector<vector<double>>);
    vector<vector<double>> Decrypt(const CVec&);
    //void readModelParams();
    //void buildLayers();

    inline int getSlots()   { return ringDim/2; }
    inline int getRingDim() { return ringDim; }
    inline int getPackingFactor() { return PF; }
    inline int calNCtxt(int n) { return ceil(float(n)/PF); }
    
    void testDecrypt(const Ciphertext<DCRTPoly>& ctx, int l=2)  {
        Plaintext ptx;
        cc->Decrypt(keyPair.secretKey, ctx, &ptx);
        auto v = ptx->GetRealPackedValue();
        cout << "  [Test] LDC: " << ctx->GetLevel() << " " << ctx->GetDepth() << " " <<  ctx->GetElements().size()
             << "  Err/Pre: " << ptx->GetLogError() << "/" << ptx->GetLogPrecision() << "    ";
        for (int p=0;p<PF;p++) cout << v[p*pinf] << " "; 
        cout  << endl;
        //ptx->SetLength(l); // caution: this will remove the rest.
    }

    private:
    void _maskCtxt(Ciphertext<DCRTPoly>& ctx, int p);

    #ifdef VERBOSE_HE
    //========================
    // Some unit test ...
    //ccTestSigmoid(keyPair);  exit(0);
    //ccTestDepth(keyPair, nMults, maxdepth);  exit(0);
    //ccTest(keyPair);  exit(0);
    //ccTestRotate(keyPair);  exit(0);
    #endif

};



#include <H5Cpp.h>
using namespace H5;

void readH5Var(const H5File& h5, string var, vector<hsize_t> & dim, vector<float> & data);


class Layer {
    public:
    string h5wvar, h5bvar;
    int in_dim, out_dim;
    int s, p;
    vector<hsize_t> wd, bd;
    vector<float> w, b;

    Layer(const H5File& h5, string h5wvar_, string h5bvar_) : h5bvar (h5bvar_), h5wvar (h5wvar_){
        readH5Var(h5, h5wvar, wd, w);
        readH5Var(h5, h5bvar, bd, b);
        info();
    }
    void info() {
       cout << "  Read dims ...  W: ";
       for(const auto& i:wd) cout << i << " "; 
       cout << "\t B: ";
       for(const auto& i:bd) cout << i << " ";
       cout << endl;
    }
    void setInDim(int in_) {  in_dim = in_;   }
    int  getInDim()        {  return in_dim;   }
    int  getOutDim()       {  return out_dim;   }
    int  getOutDim(int in_, int s_=1, int p_=2) {   // only for conv layers
        s = s_; p = p_;
        setInDim(in_);
        out_dim = ( (in_-wd[0])/s + 1 ) / p;   // we assume conv+act+pool layer
        return out_dim;
    }
};

