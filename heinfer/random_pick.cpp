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

void random_pick_fasta(const char fout[], const char fin[], int ninf = 2000) {

    srand(time(0));


    std::ifstream fhandle(fin, std::ios::in);
    if (!fhandle) {
        cout << "Cannot open " << fin << endl;
        exit(0);
    }
    
    vector<string> data;
    
    string line;
    int nmax=0, nmin=999999999;
    while (fhandle.peek() == '>') {
        
        getline(fhandle, line);
        data.emplace_back( line );

        getline(fhandle, line);
        data.emplace_back( line );

        int n = line.length();
        if (n>nmax) nmax = n;
        if (n<nmin) nmin = n;
    }
    fhandle.close();
    printf("Read   len min: %d max: %d\n", nmin, nmax);

    int insize = data.size()/2;
    if ( ninf > insize ) {
        ninf = insize;
        cout << "Data file too small. Reducing the ninf to " << ninf << endl;
    }

    //
    vector<int> indexes( insize );
    for (int i = 0; i < insize; ++i)   indexes[i] = i;
    std::random_shuffle(indexes.begin(), indexes.end());

    // Write to file
    nmax=0; nmin=999999999;
    std::ofstream out(fout, std::ios::out);
    for (int i=0; i<ninf; i++) {
        int n = data[ 2*indexes[i]+1 ].length(); 
        if (n>nmax) nmax = n;
        if (n<nmin) nmin = n;
        out << data[ 2*indexes[i] ] << endl << data[ 2*indexes[i]+1 ] << endl;
    }
    printf("Writen len min: %d max: %d\n", nmin, nmax);
    out.close();
}

int main(int argc, char* argv[]) {

    int ninf=2000;

    if (argc<3) {
        printf("Usage: %s <output.fa> <input.fa> [ninf=%d]\n", argv[0], ninf);
        return 0;
    }
    
    if (argc>3) ninf = atoi(argv[3]);
    
    random_pick_fasta(argv[1], argv[2], ninf);
    return 0;
}