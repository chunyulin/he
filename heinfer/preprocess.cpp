#include <cstdio>
#include <fstream>

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

using std::vector;
using std::string;
using std::cout;
using std::endl;

int label2int(string l) {

    vector<string> label{"0", "B.1.427", "B.1.1.7", "P.1", "B.1.526" };
    for (int i=1;i<=4;i++) {
        if (l.compare(label[i]) == 0) return i;
    }
    //cout << "** Exception lable : " << l << endl;
    return 0;
}

int kmerize(vector<double>& data, string line, int SEGLEN, int kmer, int stride) {

    // FASTA_STR="AMVHCRSNWUYBGKDT"
    int mmin = char('A')*( (1<<(8*kmer)) / (256-1) );
    int mmax = char('Y')*( (1<<(8*kmer)) / (256-1) );
    double Mm = 0.5*(mmax+mmin);
    double Md = 0.5*(mmax-mmin);

    int n = std::min( int(( line.length() - kmer )/stride+1),
                      int(( SEGLEN        - kmer )/stride+1) );

    //  simple 1d linear encoding for Kmer
    for(int d=0; d<n; d++)  {
        double enc = 0;
        for(int k = 0; k<kmer; k++) {
            enc += char(line[d*stride+k]) << ((2-k)*8);
        }
        enc = (enc-Mm)/Md;
        data[d] = enc;
    }
    return n;
}


void  preprocess_fasta(const char fout[], const char fin[], int SEGLEN, int kmer, int stride, bool readlabel = 0, int ninf = 2000) {

    std::ifstream fhandle(fin, std::ios::in);
    if (!fhandle) {
        cout << "Cannot open " << fin << endl;
        exit(0);
    }
    
    vector<int> y(ninf);
    vector<vector<double>> data(ninf, vector<double>((SEGLEN-kmer)/stride+1));
    
    string label, line;
    int nmax=0, nmin=SEGLEN;
    for (int i=0; i<ninf; i++) {
        getline(fhandle, line);
        if (readlabel)  {
            label = line.substr(1, line.find('_')-1);
            y[i] = label2int(label);
        }
        getline(fhandle, line);

        int n = kmerize(data[i], line, SEGLEN, kmer, stride);
        if (n>nmax) nmax = n;
        if (n<nmin) nmin = n;
    }
    fhandle.close();
    
    // Write to binary
    std::ofstream outbin (fout, std::ios::out | std::ios::binary);
    outbin.write (reinterpret_cast<char*>(&nmin), sizeof(nmin));
    for (int i=0; i<ninf; i++) {
        outbin.write ( reinterpret_cast<char*>(data[i].data()), nmin*sizeof(data[0][0]));
    }

    cout<< "==========="<<endl;
    for (int i=0; i<10; i++) {
        cout << data[0][i] << " ";
    }    cout<< endl << "==========="<<endl;
    
    
    outbin.close();

    cout << ninf << " sequence written. Each has "<< nmin <<" feature input." << endl;
    if (nmin != nmax)
        cout << "** Non-uniform feature input detected. Minimal feature was written." << endl;

    cout << "File size: " << (sizeof(int) + ninf*nmin*sizeof(double)) << " Bytes" << endl;

}

int main(int argc, char* argv[]) {

    int kmer=3, stride=3, SEGLEN=28500;

    if (argc<3) {
        cout << "Usage: "<< argv[0] << " <seg.bin> <fasta file> [SEGLEN=28500] [kmer=3] [offset=3]" << endl;
    }
    
    if (argc>3) SEGLEN = atoi(argv[3]);
    if (argc>4) kmer   = atoi(argv[4]);
    if (argc>5) stride = atoi(argv[5]);
    
    preprocess_fasta(argv[1], argv[2], SEGLEN, kmer, stride, 1);



}