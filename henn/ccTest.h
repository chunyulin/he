#pragma once

// Assume cc global variable

#define VERBOSE_HE


inline void testDecrypt(const LPKeyPair<DCRTPoly>& keyPair, const Ciphertext<DCRTPoly>& ctx)  {
    #ifdef VERBOSE_HE
    int nc = ctx->GetElements().size();
    Plaintext ptx;
    cc->Decrypt(keyPair.secretKey, ctx, &ptx);
    ptx->SetLength(4);
    cout <<   "  LevelDepth: " << ctx->GetLevel() << " " << ctx->GetDepth() << " " <<nc<< " " << ptx << endl;
    #endif
}


void ccTest(LPKeyPair<DCRTPoly>& keyPair) {

    Ciphertext<DCRTPoly> x,y;
    Plaintext res;
    
    const int batchSize = cc->GetRingDimension() / 2;
    vector<double> v(batchSize);
    #pragma omp parallel for
    for (int i=0; i<batchSize; i++) v[i] = 1.0;
    Plaintext ptx = cc->MakeCKKSPackedPlaintext(v);
    x = cc->Encrypt(keyPair.publicKey, ptx);


    int N=5;
    cout << "============" << endl;
    testDecrypt(keyPair, x);
    
    cout << "============" << endl;
    y = cc->EvalMult(x, x);
    testDecrypt(keyPair, y);
    for (int i=0;i<N;i++) {
        y = cc->EvalMult(y, y);
        testDecrypt(keyPair, y);
    }
    cout << "============" << endl;
    y = cc->EvalMult(x, x);
    cc->ModReduceInPlace(y);
    testDecrypt(keyPair, y);
    for (int i=0;i<N;i++) {
        y = cc->EvalMult(y, y);
        cc->ModReduceInPlace(y);
        y  = cc->LevelReduce(y, nullptr, 2);
        testDecrypt(keyPair, y);
    }
    cout << "============" << endl;
    y = cc->EvalMultNoRelin(x, x);
    cc->ModReduceInPlace(y);
    testDecrypt(keyPair, y);
    for (int i=0;i<N;i++) {
        y = cc->EvalMultNoRelin(y, y); //4
        cc->ModReduceInPlace(y);
        testDecrypt(keyPair, y);
    }
}


void ccTestDepth(LPKeyPair<DCRTPoly>& keyPair, int nMults, int maxdepth) {

    Ciphertext<DCRTPoly> x;
    Plaintext res;

    const int batchSize = 2048;
    vector<double> v(batchSize);
    #pragma omp parallel for
    for (int i=0; i<batchSize; i++) v[i] = 1.0;
    Plaintext ptx = cc->MakeCKKSPackedPlaintext(v);
    x = cc->Encrypt(keyPair.publicKey, ptx);
    const double c0 = 1.00000000000000000000;

    for (int i=0; i<nMults; i++) {
        x = cc->EvalMultNoRelin(x, x);
        x = cc->Relinearize(x);
        //x = cc->ModReduce(x);
        cc->Decrypt(keyPair.secretKey, x, &res);
        res->SetLength(5);
        cout << "Depth:" << i << " (" << x->GetLevel()<< " " << x->GetDepth() <<  ") "  << res << endl;
    }

    for (int i=0; i<maxdepth; i++) {
        x = cc->ModReduce(x);

        cc->Decrypt(keyPair.secretKey, x, &res);
        res->SetLength(5);
        cout << "Rescale:" << i << " (" << x->GetLevel()<< " " << x->GetDepth() <<  ") "  << res << endl;
    }

}

void ccTestSigmoid(LPKeyPair<DCRTPoly>& keyPair) {

    Ciphertext<DCRTPoly> x, y;
    const double c0 = 0.5;
    const double c1 = 0.25;
    const double c3 = - 1.0/48;

    const int batchSize = 2048;
    vector<double> v(batchSize);
    #pragma omp parallel for
    for (int i=0; i<batchSize; i++) v[i] = (double) (rand()-RAND_MAX/2) / (RAND_MAX);

    vector<double> exact(batchSize);
    #pragma omp parallel for
    for (int i=0; i<batchSize; i++) {
        exact[i] = 0.5 + 0.25*v[i] - 0.0208333333333333*v[i]*v[i]*v[i];
    }

    Plaintext ptx = cc->MakeCKKSPackedPlaintext(v);
    x = cc->Encrypt(keyPair.publicKey, ptx);

    #ifdef AUTORESCALE
    auto x2   = cc->EvalMultNoRelin(x, x);
    auto xo48 = cc->EvalMult(x, c3);
    auto x3   = cc->EvalMultNoRelin(x2, xo48);

    auto xo4  = cc->EvalMult(x, c1);
    cc->EvalAddInPlace(x3, xo4);
    y = cc->EvalAdd(x3, c0);
    #else
    auto x2   = cc->ModReduce( cc->EvalMultNoRelin(x, x) );
    auto xo48 = cc->ModReduce( cc->EvalMult(c3, x)       );
    auto x3   = cc->EvalMultNoRelin(xo48, x2);

    auto xo4  = cc->EvalMult(c1, x);
    cc->EvalAddInPlace(x3, xo4);
    y = cc->EvalAdd(x3, c0);
    //y = cc->Rescale(out[i]);
    #endif

    Plaintext res;
    cc->Decrypt(keyPair.secretKey, y, &res);
    //prob->SetLength(batchSize);
    //cout << prob << endl;

    auto he = res->GetCKKSPackedValue();
   
    double l2 = -999999;
    #pragma omp parallel for reduction(max:l2)
    for (int i=0; i<batchSize; i++) {
        l2 = std::max(l2, abs(exact[i] - he[i].real()) );
    }

    cout << "Max deviation: " << l2 << endl;

}

