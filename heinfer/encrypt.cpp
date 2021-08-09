#include "heinfer.h"

#include <fstream>

int label2int(string l) {
    
    vector<string> label{"0", "B.1.427", "B.1.1.7", "P.1", "B.1.526" };

    for (int i=1;i<=4;i++) {
        if (l.compare(label[i]) == 0) return i;
    }
    //cout << "** Exception lable : " << l << endl;
    return 0;
}

//
//  Read from preprocessed FASTA
//
int read_preprocessed(vector<vector<double>>& v, vector<int>& y, const char fname[]) {
     
    int ninf = 0;
     
    std::ifstream fbin(fname, std::ios::in | std::ios::binary);
    if (!fbin) {
        cout << "** Cannot open file " << fname << endl;
        exit(0);
    }
    
    // get the last dimension
    int nbp = 0;
    fbin.read (reinterpret_cast<char*>(&nbp), sizeof(nbp));

    vector<vector<double>> transpose;
    vector<double> line(nbp);
    while ( fbin.peek() != EOF ) {
        fbin.read (reinterpret_cast<char*>(line.data()), nbp*sizeof(double));
        transpose.push_back(line);
        ninf++;
    }

    vector<double> pv(ninf);
    for (int i=0; i<nbp; i++) {
        #ifdef  SANITY_CHECK
        for (int j=0; j<ninf; j++)  pv[j] = 1.0;
        #else
        for (int j=0; j<ninf; j++)  pv[j] = transpose[j][i];
        #endif
        v.push_back(pv);
    }
    
    return nbp;
}

void HEInfer::EncodeEncrypt(CVec& ctx, const vector<vector<double>>& x)  {

cout << "Dataset dimension: "<< x.size() << " x " << x[0].size() << endl;
int a;
std::cin >> a;
    #pragma omp parallel for
    for (int i=0; i<ctx.size(); i++) {

        vector<double> v(getSlots(), 0.0);
        for (int p=0; p<PF; p++) {
            
            if (i*PF+p >=nbp) continue;
            
            auto iter = x[i*PF+p].begin();
            std::copy(iter, iter+ninf, v.begin()+p*pinf);
        }
        Plaintext ptx = cc->MakeCKKSPackedPlaintext(v);
        ctx[i] = cc->Encrypt(keyPair.publicKey, ptx);
    }

}
   
