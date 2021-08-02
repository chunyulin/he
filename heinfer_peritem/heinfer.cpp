#include "heinfer.h"


void readH5Var(const H5File& h5, string var, vector<hsize_t> & dim, vector<float> & data) {
    
    DataSet dset = h5.openDataSet(var.c_str());
    
    DataSpace dspace = dset.getSpace();
    H5T_class_t type_class = dset.getTypeClass();
    hsize_t rank = dspace.getSimpleExtentNdims();
    
    dim.resize(rank); 
    
    dspace.getSimpleExtentDims(dim.data(), NULL);   // return rank=3:  fortran style. [kernel][ch][filters]
    int n = 1; for (const auto &i : dim) {  n *= i;  }

    DataSpace mspace(rank, dim.data());
    data.resize(n);
    dset.read(data.data(), PredType::NATIVE_FLOAT, mspace, dspace );
}


void ccInfo(CryptoContext<DCRTPoly>& cc) {
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
    cout << "  Q bit length: " << QBitLength << " - " << cc->GetModulus() << endl;
    auto PBitLength = P.GetLengthForBase(2);
    cout << "  P bit length: " << PBitLength << " - " << P << endl;
    cout << "  Total bit of ciphertext modulus: " << QBitLength + PBitLength << "\n\n";
}


HEInfer::HEInfer(int ninf_, int nbp_, int nMults_, int depth_, int sf_, int fmod_, int ringDim_) : 
    ninf(ninf_), nbp(nbp_), nMults(nMults_), maxdepth(depth_), scaleFactor(sf_), firstModSize(fmod_), ringDim(ringDim_) {

    printf("======= HEInfer (encryption over single item) =======\n");
    printf("  NBP  : %d\n", nbp);
    printf("  NINF : %d\n", ninf);
    printf("  nMults=%d, MaxDepth=%d, ScaleFactor=%d, DecryptBits=%d\n", nMults, maxdepth, scaleFactor, firstModSize);

    cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
            nMults, scaleFactor, 0 /*batch*/, securityLevel, ringDim,
        #ifdef AUTORESCALE
            APPROXAUTO, HYBRID,
        #else
            APPROXRESCALE, HYBRID,
        #endif
            numLargeDigits, maxdepth, firstModSize, relinWindow, OPTIMIZED);

    cc->Enable(ENCRYPTION);
    cc->Enable(SHE);
    cc->Enable(LEVELEDSHE);   // Not for BFV

    ringDim = cc->GetRingDimension();

    int nbatch = 1;
    printf("  RingDim  = %d\n", ringDim);

    //ccInfo(cc);
}

void HEInfer::genKeys(const vector<int>& rot_idx) {
    TIMER t1;

    TIC(t1);
    cout << "Generate pub/sec keys ... ";
    keyPair = cc->KeyGen();
    DURATION tKG = TOC(t1);
    cout << tKG.count() << " sec." << endl;

    TIC(t1);
    cout << "Generate multi keys ... ";
    cc->EvalMultKeyGen(keyPair.secretKey);
    DURATION tMultKG = TOC(t1);
    cout << tMultKG.count() << " sec." << endl;

    TIC(t1);
    cout << "Generate " << rot_idx.size() << " rotation keys: ";
    for (int i=0;i<rot_idx.size();i++)  cout << rot_idx[i] << " ";
    cout << "... ";
    
    cc->EvalAtIndexKeyGen(keyPair.secretKey, rot_idx);
    DURATION tRotKG = TOC(t1);
    cout << tRotKG.count() << " sec." << endl;

}

void HEInfer::test() {

    exit(0);

}

/**
 *  Conv1D:  (only support dilation = 1)
 *  Y_{n,i,f} = Pool[ \sigma ( X_{n, i+k,c}*W_{k,c,f} + B_{f} ) ]       for each {n, f, i}
 *
 *   A convolution kernel [idx= i*S*C+c, (i*S+K-1)*C+c] becomes 
 *      the sum of a few ciphertext in the packing coordinate: [ idx/PF, idx%PF ] to [ (idx+K-1)/PF, (idx+K-1)%PF ]
 *
 *  TODO: maybe optimized with better caching scheme for the packing of inference set (double packing)...
 */
void HEInfer::ConvReluAP1d(CVec& out, const CVec& in, Layer& l, const int S=1, const int P=2, const int SP=1) {

    // Need rotation key indeies
    // K1 S1 P1  -->   { 1, 2, ...,        (K1-1)      }
    // K2 S2 P2  -->   { 1, 2*S1*P1, ... , (K2-1)*S1*P1}
    // K3 S3 P3  -->   { 1, 2*S2*P2, ... , (K3-1)*S2*P2}

    cout << "Conv-Relu-Pooling layer -  ctxt in:" << in.size() << "  ctxt out:" << out.size() << " ";
    
    const int K=l.wd[0];
    const int C=l.wd[1];
    const int F=l.wd[2];
    const int I=l.getOutDim();

#ifdef SANITY_CHECK
    #pragma omp parallel for
    for (int i=0; i<K*C*F; i++) l.w[i] = 1.0/(K*C)/sqrt(P);
    #pragma omp parallel for
    for (int i=0; i<F; i++)     l.b[i] = 0.0;
#else
    // compensate rescaling model parameters for average pooling
    #pragma omp parallel for
    for (int i=0; i<K*C*F; i++) l.w[i] *= 1.0/sqrt(P);
    #pragma omp parallel for
    for (int i=0; i<F; i++)     l.b[i] *= 1.0/sqrt(P);
#endif

//TIMER t; float told = 0; TIC(t);

    #pragma omp parallel for
    for (int n=0; n<ninf; n++) {

//DURATION tt = TOC(t);
//cout << n << ": " << tt.count()-told << endl << std::flush; told = tt.count();

        vector<shared_ptr<vector<DCRTPoly>>> precomp(C);
        for (int c=0; c<C;    c++) {
            precomp[c] = cc->EvalFastRotationPrecompute(in[n*C+c]);
        }

        for (int f=0; f<F;    f++) {

#if 1
            // ~3.3 sec 
            CVec tmp(K*C);   
            #pragma omp parallel for
            for (int k=0; k<K;    k++)
            for (int c=0; c<C;    c++) {
                if (k+c==0) {
                    tmp[k*C+c] = cc->EvalMult(in[n*C+c], l.w[(k*C+c)*F+f]) ;
                } else {
                    auto shifted = cc->EvalFastRotation(in[n*C+c], k*SP, getRingDim()*2, precomp[c]);
                    tmp[k*C+c] = cc->EvalMult(shifted, l.w[(k*C+c)*F+f]) ;
                }
            }
            cc->EvalAddManyInPlace(tmp);
            cc->ModReduceInPlace(tmp[0]);
            // ReLU
            out[n*F+f] = cc->EvalMult(tmp[0], tmp[0]);
#else
            // ~5.7 sec
            Ciphertext<DCRTPoly> tmp;
            for (int k=0; k<K;    k++)
            for (int c=0; c<C;    c++) {
                if (tmp == nullptr) {
                    tmp = cc->EvalMult(in[n*C+c], l.w[(k*C+c)*F+f]) ;
                } else {
                    auto shifted = cc->EvalFastRotation(in[n*C+c], k*SP, getRingDim()*2, precomp[c]);
                    cc->EvalAddInPlace(tmp, cc->EvalMult(shifted, l.w[(k*C+c)*F+f]) );
                }
            }
            cc->ModReduceInPlace(tmp);
            // ReLU
            out[n*F+f] = cc->EvalMult(tmp, tmp);
#endif

            cc->ModReduceInPlace(out[n*F+f]);

            // Avg Pooling
            for (int p=1; p<P; p++) {
                cc->EvalAddInPlace( out[n*F+f], cc->EvalAtIndex(out[n*F+f], p*SP) );
            }
        }
    }
}

/**
 *  Dense:
 *  Y_i = X_j*W_ji * + B_i
 *
 */
void HEInfer::DenseSigmoid(CVec& out, const CVec& in, Layer& l, const int C=1, const int SP=1) {

    const int J=l.getInDim();
    const int I=l.wd[1];

    vector<Plaintext> LB(I);
    for (int i=0;i<I;i++)  {
        vector<double> one_( getSlots(), l.b[i] );
        LB[i] = cc->MakeCKKSPackedPlaintext(one_, 1);
    }

    int nodes = J/C;
    printf("Dense layer : %d x %d, active nodes per ctxt = %f\n", J, I, nodes);
    if (J%C>0) cout << "**** Inconsistent nodes in each ctx !  (" << J/float(C) << " not an integral.)" << endl;

#ifdef  SANITY_CHECK
    #pragma omp parallel for
    for (int i=0; i<I*J; i++) l.w[i] = 1.0/(J);
    #pragma omp parallel for
    for (int i=0; i<I; i++)     l.b[i] = 0.0;
#endif

    //const double c0 = 0.5,  c1 = 0.25,     c3 = - 1.0/48;   // Taylor expansion
    const double   c0 = 0.5,  c1 = - 1.2/8.0,  c3 = 0.81562/8.0;
  
    #pragma omp parallel for
    for (int n=0; n<ninf; n++)
    for (int i=0; i<I; i++) {

        CVec tmp(C);
        for (int c = 0; c<C;  c++) {

            vector<double> wv( getSlots(), 0.0);
            for (int s = 0; s<nodes; s++)  wv[s*SP] = l.w[(s*C+c)*I+i];
            Plaintext w = cc->MakeCKKSPackedPlaintext(wv, 1);

            tmp[c] = cc->EvalMult( in[n*C+c], w );
        }

        int oi = n*I+i;

        cc->EvalAddManyInPlace(tmp);
        out[oi] = cc->EvalAdd(LB[i], tmp[0]);
        cc->ModReduceInPlace(out[oi]);

#ifndef SIGMOID_ORD_1
        // Another expression in [-8,8]:  0.5 - 1.2/8 * x + 0.81562/8 * x**3
        // Taylor expansion Sigmoid: 0.5 + 0.25*x - x**3/48

        auto x2   = cc->EvalMult(out[oi], out[oi]);    // d2, c3
        cc->ModReduceInPlace( x2 );
        auto xo48 = cc->EvalMult(out[oi], c3);
        cc->ModReduceInPlace( xo48 );
        auto x3   = cc->EvalMultNoRelin(x2, xo48);

        auto xo4  = cc->EvalMult(out[oi], c1);
        cc->EvalAddInPlace(x3, xo4);
        cc->ModReduceInPlace(x3);
        out[oi] = cc->EvalAdd(x3, c0);
#else
        cout<< "Not implement yet." << endl;
#endif
    }
}


vector<vector<double>> HEInfer::Decrypt(const CVec& c) {

    vector<vector<double>> v(c.size(), vector<double>(ninf));
    
    #pragma omp parallel for
    for (int i=0; i<c.size(); i++) {
        Plaintext prob;
        cc->Decrypt(keyPair.secretKey, c[i], &prob);
        v[i/ninf][i%ninf] = prob->GetRealPackedValue()[0];
    }
    return v;
}