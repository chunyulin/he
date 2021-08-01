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

void preprocess_fasta(vector<vector<double>>& v, vector<int>& y, const char fname[]) {

    // Lookup table for FASTA letters c: g2v[char(c)-45] = (c-len(FASTA_STR)//2)/len(FASTA_STR)*2
    // FASTA_STR = "AMVHCR-SNWUYBGKDT";
    const double g2v[] = {-0.23529411764705882 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-0.9411764705882353 ,0.47058823529411764 ,-0.47058823529411764 ,0.8235294117647058 ,0.0 ,0.0 ,0.5882352941176471 ,-0.5882352941176471 ,0.0 ,0.0 ,0.7058823529411765 ,0.0 ,-0.8235294117647058 ,0.0 ,0.0 ,0.0 ,0.0 ,-0.35294117647058826 ,-0.11764705882352941 ,0.9411764705882353 ,0.23529411764705882 ,-0.7058823529411765 ,0.11764705882352941 ,0.0 ,0.35294117647058826};

    int nbp  = v.size();
    int ninf = v[0].size();

    std::fstream fasta_file(fname, std::ios::in);
    if (!fasta_file) {
        cout << "Cannot open file!" << endl;
        exit(0);
    }
    string label, line;
    for (int i=0; i<ninf; i++) {
        getline(fasta_file, line);
        label = line.substr(1, line.find('_')-1);
        y[i] = label2int(label);
        getline(fasta_file, line);
        #ifdef  SANITY_CHECK
        for (int j=0; j<nbp; j++)  v[j][i] = 1.0;
        #else
        for (int j=0; j<nbp; j++)  v[j][i] = g2v[int(line[i])-45];
        #endif
    }
}

void HEInfer::EncodeEncrypt(CVec& ctx, const vector<vector<double>> x)  {

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
   
