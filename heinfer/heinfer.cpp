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


HEInfer::HEInfer(int ninf_, int nbp_, int nMults_, int depth_, int sf_, int fmod_, int ringDim_, int pinf_ = 0) : 
    ninf(ninf_), nbp(nbp_), nMults(nMults_), maxdepth(depth_), scaleFactor(sf_), firstModSize(fmod_), ringDim(ringDim_) {

    printf("Initialized Palisade Cryptocontext...\n");
    printf("   NBP  : %d\n", nbp);
    printf("   NINF : %d\n", ninf);
    printf("   nMults=%d, MaxDepth=%d, ScaleFactor=%d, DecryptBits=%d\n", nMults, maxdepth, scaleFactor, firstModSize);

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
    if (pinf_ == 0)   pinf = 1 << int( std::ceil(std::log2(ninf)) );
    else              pinf = pinf_;
    PF = (ringDim/pinf)>>1;
    
    int nbatch = 1;
    printf("   RingDim  = %d\n", ringDim);
    printf("   Infer packing = %d   Packing factor = %d   (Slot util.: %.1f%)\n\n", pinf, PF, float(ninf)/pinf*100);

    //ccInfo(cc);

    // Prepare masking plaintext
    for (int p=0; p<PF; p++) {
        vector<double> v(getSlots(),0);
        std::fill (v.begin()+p*pinf, v.begin()+p*pinf+ninf, 1.0);
        _mask.emplace_back(cc->MakeCKKSPackedPlaintext(v, 1));
    }
}

void HEInfer::genKeys() {
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

    if (PF > 1 ) {
        TIC(t1);
        cout << "Generate " << PF-1 << " rotation keys ... ";

        vector<int> ilist(PF-1);
        #pragma omp parallel for
        for (int i=0;i<PF-1;i++) ilist[i] = (i+1)*pinf;   // + for rotating to the left

        cc->EvalAtIndexKeyGen(keyPair.secretKey, ilist);
        DURATION tRotKG = TOC(t1);
        cout << tRotKG.count() << " sec." << endl;
    }
}

void HEInfer::test() {

    vector<double> v( getSlots(), 0.0);
    for (int i=0; i<getSlots(); i++) {
        v[i] = 1.0;
    }
    Plaintext x =  cc->MakeCKKSPackedPlaintext(v, 1);
    auto ctx = cc->Encrypt(keyPair.publicKey, x);

    for(int j = pinf; j < getSlots(); j*=2)  {
        cc->EvalAddInPlace(ctx, cc->EvalAtIndex(ctx, j));
    }
    testDecrypt(ctx);
    exit(0);

}

void HEInfer::_maskCtxt(Ciphertext<DCRTPoly>& ctx, int p) {
    //if (op!=0) x[0] = cc->EvalAtIndex(x[0], (PF-op)*pinf); // Rotation seems introduce too much noise!?!
    ctx = cc->EvalMult(ctx, _mask[p]);
    cc->RescaleInPlace(ctx);
}


/**
 *  Conv1D:  (only support dilation = 1)
 *  Y_{i,f} = Pool[ \sigma ( X_{S*i+k,c}*W_{k,c,f} + B_{f} ) ]       for each {i, f}
 *
 *   A convolution kernel [idx= i*S*C+c, (i*S+K-1)*C+c] becomes 
 *      the sum of a few ciphertext in the packing coordinate: [ idx/PF, idx%PF ] to [ (idx+K-1)/PF, (idx+K-1)%PF ]
 *
 *  TODO: maybe optimized with better caching scheme for the packing of inference set (double packing)...
 */
void HEInfer::ConvReluAP1d(CVec& out, const CVec& in, Layer& l, const int S=1, const int P=2) {

    cout << "Conv-Relu-Pooling layer -  ctxt in:" << in.size() << "  ctxt out:" << out.size() << " ";

    const int K=l.wd[0];
    const int C=l.wd[1];
    const int F=l.wd[2];
    const int I=l.getOutDim();

//cout << endl << "w[F]: ";for (int f=0;f<F;f++) printf("%.10f  " , l.w[f]);cout << endl;
//cout << endl << "b[F]: ";for (int f=0;f<F;f++) printf("%.10f  " , l.b[f]);cout << endl;

#ifdef SANITY_CHECK_MODEL
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

    // Convolution index:  starting from [i*S*C+c]
    // for-loop over each output neuron and filters
    #pragma omp parallel for
    for (int ic = 0; ic<out.size(); ic++) {    // (physcial index on packing space) index for out-ciphertext
      for (int ip = 0; ip<PF;         ip++) {  // packing index in the out-ciphertext. Not parallelable.

        int i = (ic*PF+ip)/F;    // logical output indices
        int f = (ic*PF+ip)%F;

        if (ic*PF+ip >= I*F) continue;

        CVec x(P);

        // for-loop over pooling neuron
        #pragma omp parallel for
        for (int p=0; p<P; p++) {

            vector<double> w( getSlots(), 0.0);

            for (int k=0; k<K; k++)   // Note: Can this part go parallel ?
            for (int c=0; c<C; c++) {
                int kl = ((i*P+p)*S+k)*C+c;
                int kc = kl/PF;        // which ciphertext
                int kp = kl%PF;        // which packing index in the ciphertext

//cout<<  l.w[(k*C+c)*F+f] << "==========="<< i<< " " << f << " "<< p << " " << kc << " " << kp << endl;

                auto iter = w.begin() + kp*pinf;
                std::fill (iter, iter+ninf, l.w[(k*C+c)*F+f]);

                // once complete a full package, comp W*X ciphertext and save to convSum
                if ( kp == PF-1 || (k*C+c == C*K-1) ) {
                    Plaintext wenc = cc->MakeCKKSPackedPlaintext(w, 1);
                    if (x[p] == nullptr) x[p] = cc->EvalMult(in[kc], wenc);
                    else                 cc->EvalAddInPlace(x[p], cc->EvalMult(in[kc], wenc));
                    std::fill(w.begin(), w.end(), 0);
                }

            }   // end of partial conv sum

            //--- Conv sum
            for(int j = pinf; j < getSlots(); j*=2)  {
                cc->EvalAddInPlace(x[p], cc->EvalAtIndex(x[p], j));
            }
            x[p] = cc->EvalAdd(x[p], l.b[f]);

            cc->ModReduceInPlace( x[p] );

            // ReLU (x^2)
            x[p] = cc->EvalMultNoRelin(x[p], x[p]);
        }  // end of p

        // Average pool
        cc->EvalAddManyInPlace(x);
        _maskCtxt(x[0], ip);
        cc->ModReduceInPlace( x[0] );

//cout <<"========="; testDecrypt(x[0]);

        #pragma omp critical
        {
        if ( out[ic] == nullptr ) out[ic] = x[0];
        else                      cc->EvalAddInPlace(out[ic], x[0]);
        }
        //if (ip==0&&ic%10==0) cout << "." << std::flush;
      
      }

      cc->RelinearizeInPlace( out[ic] );

    }  // end of output ciphter text (ic)

    cout << endl;
}

/**
 *  Conv1D:  (only support dilation = 1)
 *  Y_{i,f} = Pool[ \sigma ( X_{S*i+k,c}*W_{k,c,f} + B_{f} ) ]       for each {i, f}
 */
//void HEInfer::ConvReluAP1d_plain(vector<vector<double>>&in, Layer& l, const int S=1, const int P=2) {
void __ConvReluAP1d_plain(vector<vector<double>>&in, Layer& l, const int S=1, const int P=2) {

    const int K=l.wd[0];
    const int C=l.wd[1];
    const int F=l.wd[2];
    const int I=l.getOutDim();

    for (int f = 0; f<F; f++) {
      for (int i = 0; i<I; i++) {

        vector<double> x(P,0);
        
        #pragma omp parallel for
        for (int p=0; p<P; p++) {

            x[p] =  1;
            for (int k=0; k<K; k++)
            for (int c=0; c<C; c++) {
                int kl = ((i*P+p)*S+k)*C+c;
                x[p] += in[i][0]*l.w[(k*C+c)*F+f];

            }   // end of partial conv sum

            // ReLU (x^2)
            x[p] = x[p]*x[p];
        }
      }
    }
}


/**
 *  Dense:
 *  Y_i = X_j*W_ji * + B_i
 *
 */
void HEInfer::DenseSigmoid(CVec& out, const CVec& in, Layer& l) {

    const int J=l.getInDim();
    const int I=l.wd[1];
    printf("Dense layer : %d x %d,     ctxt ( %d x %d )\n", J, I, in.size(), out.size());

//cout << endl << "w[F]: ";for (int f=0;f<J;f++) printf("%.6f  " , l.w[f]);cout << endl;
//cout << endl << "b[F]: ";for (int f=0;f<I;f++)  printf("%.6f  " , l.b[f]);cout << endl;

#ifdef  SANITY_CHECK_MODEL
    #pragma omp parallel for
    for (int i=0; i<I*J; i++) l.w[i] = 1.0/(J);
    #pragma omp parallel for
    for (int i=0; i<I; i++)     l.b[i] = 0.0;
#endif

    const double   c0 = 0.5,  c1 = - 1.2/8.0,  c3 = 0.81562/8.0;

    #pragma omp parallel for
    for (int i = 0; i<out.size();  i++) {         // index of out-ciphertext

        CVec x(in.size());
        #pragma omp parallel for
        for (int jb = 0; jb<in.size();  jb++) {   // loop over in-ciphertext (big-jump over PF)

            vector<double> wv( getSlots(), 0.0);
            #pragma omp parallel for
            for (int jp = 0; jp<PF; jp++) {
                if (jb*PF+jp < J) std::fill (wv.begin()+jp*pinf, wv.begin()+jp*pinf+ninf, l.w[(jb*PF+jp)*I+i]);
            }
            Plaintext w = cc->MakeCKKSPackedPlaintext(wv, 1);

            x[jb] = cc->EvalMult( in[jb], w );
        }

        cc->EvalAddManyInPlace(x);

        // Sum over packing group
        for(int j = pinf; j < getSlots(); j*=2)  {
            cc->EvalAddInPlace(x[0], cc->EvalAtIndex(x[0], j) );
        }
        
        out[i] = cc->EvalAdd( x[0], l.b[i]);
        cc->ModReduceInPlace( out[i] );

#ifndef SIGMOID_ORD_1
        // Another expression in [-8,8]:  0.5 - 1.2/8 * x + 0.81562/8 * x**3
        // Taylor expansion Sigmoid: 0.5 + 0.25*x - x**3/48


        auto x2   = cc->EvalMult(out[i], out[i]);   cc->ModReduceInPlace( x2 );
        auto xo48 = cc->EvalMult(out[i], c3);     cc->ModReduceInPlace( xo48 );
        auto x3   = cc->EvalMult(x2, xo48);

        auto xo4  = cc->EvalMult(out[i], c1);
        cc->EvalAddInPlace(x3, xo4);
        cc->ModReduceInPlace(x3);
        out[i] = cc->EvalAdd(x3, c0);

#else
        cout<< "Not implement yet." << endl;
#endif

    }
}

/**
 *  Dense:   (Not used)
 *  Y_i = X_j*W_ji * + B_i
 *
 */
void HEInfer::DenseSigmoid_packed(CVec& out, const CVec& in, Layer& l) {

    cout << "Not implement yet!" << endl; exit(0);

    const int J=l.getInDim();
    const int I=l.wd[1];
    printf("Dense layer : %d x %d,     ctxt ( %d x %d )\n", J, I, in.size(), out.size());

#ifdef  SANITY_CHECK_MODEL
    #pragma omp parallel for
    for (int i=0; i<I*J; i++) l.w[i] = 0.5/(J);
    #pragma omp parallel for
    for (int i=0; i<I; i++)     l.b[i] = 0.5;
#endif

    const double c0 = 0.5;
    const double c1 = 0.25;
    const double c3 = - 1.0/48;

    #pragma omp parallel for
    for (int ic = 0; ic<out.size();  ic++)
    for (int ip = 0; ip<PF;          ip++)   {  // inner loop not parallelable


        int i = (ic*PF+ip);
        if (i >= I) continue;

{ cout <<i<< "==" << ic<< " " << ip << endl; };

        Ciphertext<DCRTPoly> x;
        
        for (int jb = 0; jb<J;  jb+=PF) {
        
            vector<double> wv( getSlots(), 0.0);
            #pragma omp parallel for
            for (int jp = 0; jp<PF; jp++) {
                if (jb+jp >= J) continue;
                std::fill (wv.begin()+jp*pinf, wv.begin()+(jp+1)*pinf, l.w[(jb+jp)*I+i]);
            }
            Plaintext w = cc->MakeCKKSPackedPlaintext(wv, 1);

            if (x == nullptr)  x = cc->EvalAdd( cc->EvalMult( in[jb], w), l.b[i] );
            else               cc->EvalAddInPlace(x,  cc->EvalMult( in[jb], w));
        }

        // Sum over packing group
        for(int j = pinf; j < getSlots(); j*=2)  {
            cc->EvalAddInPlace(x, cc->EvalAtIndex(x, j) );
        }


#ifndef SIGMOID_ORD_1
        // Another expression in [-8,8]:  0.5 - 1.2/8 * x + 0.81562/8 * x**3
        // Taylor expansion Sigmoid: 0.5 + 0.25*x - x**3/48
        auto x2   = cc->EvalMultNoRelin(x, x);    // d2, c3
        cc->ModReduceInPlace( x2 );
        cc->ModReduceInPlace( x2 );
        auto xo48 = cc->EvalMult(x, c3);
        auto x3   = cc->EvalMultNoRelin(x2, xo48);
        cc->ModReduceInPlace(x3);
        cc->ModReduceInPlace(x3);

        auto xo4  = cc->EvalMult(x, c1);
        cc->EvalAddInPlace(x3, xo4);
        cc->ModReduceInPlace(x3);
        out[ic] = cc->EvalAdd(x3, c0);
#else
        const double c1 = 0.15625;
#endif

        _maskCtxt(x, ip);
        if ( out[ic] == nullptr ) out[ic] = x;
        else                      cc->EvalAddInPlace(out[ic], x);
    }
}

vector<vector<double>> HEInfer::Decrypt(const CVec& c) {

    vector<vector<double>> v(c.size(), vector<double>(ninf));
    #pragma omp parallel for
    for (int i=0; i<c.size(); i++) {
        Plaintext prob;
        cc->Decrypt(keyPair.secretKey, c[i], &prob);
        auto vc = prob->GetRealPackedValue();
        for(int j=0; j<ninf; j++) v[i][j] = vc[j];

        //#pragma omp critical
        //cout << "   Err/Pre: " << prob->GetLogError() << " / " << prob->GetLogPrecision() << endl;

        #if 0
        //  why not work properly??
        auto iter = prob->GetRealPackedValue().begin();
        std::copy(iter, iter+ninf, v[i].begin());
        #endif

    }
    return v;
}