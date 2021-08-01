#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include <cstdio>
#include <H5Cpp.h>
using namespace H5;

typedef float real;
const char* H5NAME = "CNN_letter.weights.final.h5";

void readH5Var(const H5File& h5, string var, vector<hsize_t> & dim, vector<real> & data) {
    
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

class Layer {
    public:
    string h5wvar, h5bvar;
    vector<hsize_t> wd, bd;
    vector<float> w, b;

    Layer(const H5File& h5, string h5wvar_, string h5bvar_) : h5bvar (h5bvar_), h5wvar (h5wvar_){
        readH5Var(h5, h5wvar, wd, w); 
        readH5Var(h5, h5bvar, bd, b);
    }
};

int main (void)
{
    H5File h5d(H5NAME, H5F_ACC_RDONLY);
    Layer conv1 (h5d, "/conv1/conv1/kernel:0", "/conv1/conv1/bias:0");
    Layer conv2 (h5d, "/conv2/conv2/kernel:0", "/conv2/conv2/bias:0");
    Layer conv3 (h5d, "/conv3/conv3/kernel:0", "/conv3/conv3/bias:0");
    Layer output(h5d, "/output/output/kernel:0", "/output/output/bias:0");

    for (int i=0; i<160; i++) {
        cout << conv1.w[i] << " ";
    }


}


