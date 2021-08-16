#include "utils.h"
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

auto ipow = [] (int X, int k) { 
     int result = 1;
     for(int i = 0; i < k; ++i) result *= X;
     return result; 
};

int kmerize_miniencode(vector<double>& data, string line, int SEGLEN, int kmer, int stride) {

    string FASTA_STR="AMVHCRSNWUYBGKDT";
    int len  = FASTA_STR.length();
    int dmax = len-1;
    double vmax = 1.0/(ipow(len,kmer)-1);
    int n = std::min( int(( line.length() - kmer )/stride+1),
                      int(( SEGLEN        - kmer )/stride+1) );

    //  simple 1d linear encoding for Kmer
    for(int d=0; d<n; d++)  {
        double x = 0;
        for(int k = 0; k<kmer; k++) {
            x = x*len + FASTA_STR.find_first_of( line[d*stride+k] );
        }
        data[d] = 2.0*x*vmax - 1.0;
    }
    return n;
}

vector<int> read_imask(const char fname[])  {

    // read mask
    std::ifstream fmask(fname, std::ios::in);
    if (!fmask) {
        cout << "Cannot open " << fname << endl;
        exit(0);
    }
    
    string mask;
    getline(fmask, mask);
    fmask.close();
    
    const int mlen = mask.length();
    vector<int> imask(mlen);
    for (int i=0;i<mlen; i++) {
        imask[i] = ( int(mask[i]) == char('1') );
    }
    return imask;
}

void applymask(string& str, const vector<int>& imask, const int SEGLEN) {

    int mlen = imask.size();
    int ilen = str.length();

    if (ilen<SEGLEN) {
        cout<< "SEGLEN too small!" << endl;
    }

    string copy(str);
    
    int i = 0;
    for (int j=0; j<mlen; j++) {
        if (imask[j]==1) {
            str[i] = copy[j];
            i++;
            if (i>=SEGLEN) break;
        }
    }
    str.resize(SEGLEN);

}


void  preprocess_fasta(const char fout[], const char fin[], int SEGLEN, const char mask_file[], int kmer = 3, int stride = 3, bool readlabel = 0, int ninf = 2000) {

    
    vector<int> imask = read_imask(mask_file);
    
    // read fasta file 
    std::ifstream fhandle(fin, std::ios::in);
    if (!fhandle) {
        cout << "Cannot open " << fin << endl;
        exit(0);
    }
    
    vector<int> y(ninf);
    vector<vector<double>> data(ninf, vector<double>((SEGLEN-kmer)/stride+1));
    
    string label, line;
    int nmax=0, nmin=SEGLEN;
    int imax=0, imin=SEGLEN;
    for (int i=0; i<ninf; i++) {
        
        getline(fhandle, line);
        if (readlabel)  {
            label = line.substr(1, line.find('_')-1);
            y[i] = label2int(label);
        }

        getline(fhandle, line);
        applymask(line, imask, SEGLEN);
    
        int n = kmerize_miniencode(data[i], line, SEGLEN, kmer, stride);

        if (n>nmax) nmax = n;
        if (n<nmin) nmin = n;

        int in = line.length();
        if (in>imax) imax = in;
        if (in<imin) imin = in;
    }
    fhandle.close();

    #if 0
    // check min max
    double vmax=-9999, vmin=9999, sum=0;
    #pragma omp parallel for collapse(2) reduction(min:vmin) reduction(max:vmax) reduction(+:sum)
    for (int i=0; i<ninf; i++)
    for (int j=0; j<nmin; j++) {
        if (data[i][j] > vmax) vmax = data[i][j];
        if (data[i][j] < vmin) vmin = data[i][j];
        sum+= data[i][j];
    }
    printf("Encoding range: %.8f %.8f   Sum: %.8f\n", vmin, vmax, sum);

    {   // sanity check
    vector<double> test(4);
    string l = "ACGTACGTACGT";
    kmerize_miniencode(test, l, 12, 3, 3);
    for (int j=0;j<4;j++) printf(" %c%c%c (%.8f)", l[j*3], l[j*3+1], l[j*3+2], test[j]);
    cout << endl;
    }
    #endif

    // Write to binary
    std::ofstream outbin (fout, std::ios::out | std::ios::binary);
    outbin.write (reinterpret_cast<char*>(&nmin), sizeof(nmin));
    for (int i=0; i<ninf; i++) {
        outbin.write ( reinterpret_cast<char*>(data[i].data()), nmin*sizeof(data[0][0]));
    }
    outbin.close();

    cout << "Input bp min: " << imin << "  max:" << imax << endl;

    if (readlabel)  {
        // Write label
        std::ofstream flabel ("label.txt", std::ios::out | std::ios::trunc);
        for (int i=0; i<ninf; i++) { flabel << y[i] << endl; }
        flabel.close();
        cout << "Label.txt written." << endl;
    }

    cout << ninf << " sequence written. Each has "<< nmin <<" feature input." << endl;
    if (nmin != nmax)
        cout << "** Non-uniform numbers of input features detected. Minimal number was written." << endl;

    cout << "File size: " << (sizeof(int) + ninf*nmin*sizeof(double)) << " Bytes" << endl;

}

int main(int argc, char* argv[]) {

    int kmer=3, stride=3, SEGLEN=28500, ninf=2000;
    int label = 0;

    if (argc<3) {
        printf("Usage: %s <output.bin> <fasta file> [readlabel=%d] [SEGLEN=%d] [maskfile=%d] [ninf=%d] [kmer=%d] [stride=%d]\n",
           argv[0], label, SEGLEN, "mask28921.dat", ninf, kmer, stride);
        return 0;
    }
    
    if (argc>3) label  = atoi(argv[3]);
    if (argc>4) SEGLEN = atoi(argv[4]);
    //if (argc>5) offset = atoi(argv[5]);
    if (argc>6) ninf   = atoi(argv[6]);
    if (argc>7) kmer   = atoi(argv[7]);
    if (argc>8) stride = atoi(argv[8]);
    
    cout << "Preprocessing with mask file: " << "mask28921.dat" << endl;
    preprocess_fasta(argv[1], argv[2], SEGLEN, "mask28921.dat", kmer, stride, label, ninf);


    double tmem = getMemoryUsage();
    printf("Memory usage (MB): %.2f\n", tmem/1024.0);

}