#include<fstream>
#include <helib/helib.h>

#include <iostream>
#include <omp.h>
#include <iomanip>
#include <complex>
using std::complex;


/// For performance profiling.
#include <chrono>
typedef  std::chrono::duration<double> DURATION;
#undef TIC
#undef TOC
#define TIC(t) t = std::chrono::system_clock::now();
#define TOC(t) (std::chrono::system_clock::now() - t);

std::chrono::time_point<std::chrono::system_clock> t_tmp, t_global, t;

DURATION t_pk, t_sk, t_rk, t_gk;        /// key related.
DURATION t_input, t_enc, t_dec, t_square, t_batch_sum, t_reduct_sum;   /// computation related.
DURATION t_raw, t_total;

using namespace std;
using namespace helib;

int main(int argc, char* argv[])
{

    int num_batch   = 10;
    int scaleFactor = 39;    // 39
    int batchSize = 8192;
    int bits = 239;
    int nDepth = 2;
    
    if (argc > 1) num_batch   = atoi(argv[1]);
    if (argc > 2) batchSize   = atoi(argv[2]);
    if (argc > 3) scaleFactor = atoi(argv[3]);
    if (argc > 4) bits        = atoi(argv[4]);
    if (argc > 5) nDepth      = atoi(argv[5]);
    
    int ncyclo = batchSize*4;

    srand(time(0));

    // Get the num of cores.
    int num_cores   = omp_get_max_threads();
    int num_threads = omp_get_num_threads();

    cout << "==== # of cores   : " << num_cores   << " ==== " << endl;
    cout << "==== # of threads : " << num_threads << " ==== " << endl;

    TIC(t_global);

    //omp_set_num_threads(num_threads);
    omp_set_num_threads(num_cores);

    cout << "Current OpenMP #threads : " << omp_get_num_threads() << " ==== " << endl;

    // initialize a Context object using the builder pattern
    Context context =
        ContextBuilder<CKKS>().m(ncyclo).bits(bits).precision(scaleFactor).c(nDepth).build();

    // We can print out the estimated security level.
    // This estimate is based on the LWE security estimator.
    cout << "securityLevel=" << context.securityLevel() << "\n";

    // Get the number of slots, n.  Note that for CKKS, we always have n=m/4.
    // per batch
    int slot_count = context.getNSlots();

    cout << " ==== Elapsed time for creating HElib Context ==== " << endl;
    cout << " ================================================= " << endl;

    // Prepare keys.
    TIC(t_tmp);
    SecKey secretKey(context);
    secretKey.GenSecKey();
    t_sk = TOC(t_tmp); 

    // In HElib, the SecKey class is actually a subclass if the PubKey class.  So
    // one way to initialize a public key object is like this:
    TIC(t_tmp);
    const PubKey& publicKey = secretKey;
    t_pk = TOC(t_tmp);

    //RelinKeys relin_keys;
    TIC(t_tmp);
    t_rk = TOC(t_tmp);

    //GaloisKeys gal_keys;
    TIC(t_tmp);
    addSome1DMatrices(secretKey);
    t_gk = TOC(t_tmp);

    cout << "==== Number of slots: " << slot_count << endl;
    cout << "==================================================" << endl;

    int N = batchSize*num_batch;
    std::vector<double> x(N);

#pragma omp parallel for
    for (int i=0; i<N; i++) x[i] = (double) (rand()-RAND_MAX/2) / (RAND_MAX);


    cout << "# of data  : " << N << endl;
    cout << "num_batch  : " << num_batch << endl;
    cout << "slot_count : " << slot_count << endl;

    TIC(t_tmp);
    double he_result = 0.0;
    double raw_result = 0.0;
    vector<double> result_vec(slot_count);
    /// calculate the raw result.
#pragma omp parallel for reduction(+ : raw_result)
    for(int i=0; i < x.size(); i++){
        raw_result += x[i]*x[i];
    }
    t_raw = TOC(t_tmp);

    /// Encode & Encrypt.
    vector<Ctxt> c0;
    unsigned int batch_sz = slot_count;

    TIC(t_tmp);
    #pragma omp parallel for 
    for (int i=0; i< num_batch; i++){
        vector<double> x_batch = {x.begin() + i*batch_sz, x.begin() + (i+1)*batch_sz};
        PtxtArray p0(context, x_batch);
        c0.emplace_back(publicKey);
        p0.encrypt(c0[i]);
    }
    t_enc = TOC(t_tmp);

    cout << "Compute x^2 and relinearize..." << endl;

    /// sum the whole slot with rotations.
    TIC(t_tmp);
    c0[0].multLowLvl(c0[0]);
    //Ctxt_array[0].reLinearize();
    for (int i=1; i < num_batch; i++){
        c0[i].multLowLvl(c0[i]);
        //Ctxt_array[i].reLinearize();
        c0[0] += c0[i];
    }
    t_batch_sum = TOC(t_tmp);

    TIC(t_tmp);
    c0[0].reLinearize();
    totalSums(c0[0]);
    t_reduct_sum = TOC(t_tmp);

    //Plaintext plain_result;
    PtxtArray plain_result(context);

    TIC(t_tmp);
    plain_result.decrypt(c0[0], secretKey);
    plain_result.store(result_vec);
    he_result = result_vec[0];
    t_dec = TOC(t_tmp);

    /// time for whole calculation.
    t_total = TOC(t_global);

    //cout << "Decrypted   : " << t_dec.count() << endl;
    double error = he_result - raw_result;
    double rerr  = (he_result - raw_result)/raw_result;

    printf( "CKKS Result : %.6f\n", he_result);
    printf( "Raw  Result : %.6f\n", raw_result);
    printf( "Rel Error (%) : %.3g\n\n", rerr*100);

    cout << "======== Time in secs ========" << endl;
    cout << "Public Key        : " << t_pk.count() << endl;
    cout << "Secret Key Gen    : " << t_sk.count() << endl;
    cout << "Rotation Key      : " << t_gk.count() << endl;
    cout << "Relinearilzation  : " << t_rk.count() << endl << endl;

    cout << "Prepare Input     : " << t_input.count() <<endl;
    cout << "Encode/Encrypt    : " << t_enc.count() <<endl;
    cout << "Decode/Decrypt    : " << t_dec.count() <<endl;
    cout << "Compute  per b    : " << t_batch_sum.count() / num_batch <<endl;
    cout << "EvalSum           : " << t_reduct_sum.count()  <<endl;

    cout << "Total: " << t_total.count()  <<endl;

    cout << "[HElib_Summary] " << num_batch  << " "
        << t_pk.count() + t_sk.count() << " " << t_gk.count()<< " " << t_rk.count() << " "
        << t_enc.count()   << " " << t_dec.count() << " " << t_batch_sum.count()<< " " <<  t_reduct_sum.count() << " "
        << t_total.count() << " " << t_raw.count() << " " << error << " "
        << num_batch       << " " << batchSize     << " " <<  scaleFactor << " " << bits << " " << rerr << endl;

    return 0;
}

