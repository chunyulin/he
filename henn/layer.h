#pragma once
typedef float real;

#include <iostream>
#include <string>
#include <vector>
using std::vector;
using std::string;
using std::endl;
using std::cout;

#include <H5Cpp.h>
using namespace H5;


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
       cout << "  Read dims..  W: ";
       for(const auto& i:wd) cout << i << " "; 
       cout << "    B: ";
       for(const auto& i:bd) cout << i << " ";
       cout << endl;
    }
    void setInDim(int in_) {  in_dim = in_;   }
    int  getInDim()        {  return in_dim;   }
    int  getOutDim()       {  return out_dim;   }
    int  getOutDim(int in_, int s_=1, int p_=2) {
        s = s_; p = p_;
        setInDim(in_);
        out_dim = ( (in_-wd[0])/s + 1 ) / p;   // we assume conv+act+pool layer
        return out_dim;
    }
};

