#include<fstream>
#include "seal/seal.h"
//#include "examples.h"

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
using namespace seal;

int main(int argc, char* argv[])
{
    srand(time(0));
    cout << "# of Cores        : " << omp_get_max_threads() << endl;
    
    int num_batch = 1;
    int batchSize = 8192;
    int sf = 39;
    int FIRSTBIT = 60;

    if (argc > 1) num_batch = atoi(argv[1]);
    if (argc > 2) batchSize = atoi(argv[2]);
    if (argc > 3) sf        = atoi(argv[3]);
    if (argc > 4) FIRSTBIT  = atoi(argv[4]);

    double scale = pow(2.0, sf);

    TIC(t_global);

    // Encryption parameters.
    EncryptionParameters parms(scheme_type::ckks);

    // Set prms for polynomial ring.
    size_t poly_modulus_degree = 8192*2;
    parms.set_poly_modulus_degree(poly_modulus_degree);

    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { FIRSTBIT, sf, sf, FIRSTBIT }));
    //auto p128 = seal::util::global_variables::GetDefaultCoeffModulus128();
    //parms.set_coeff_modulus( p128[16384] );

    SEALContext context(parms);
    //print_parameters(context);
    //cout << endl;

    cout << " ==== Elapsed time for creating SEALContext ==== " << endl;
    cout << "=================================================" << endl;

    // Prepare keys.
    KeyGenerator keygen(context);
    TIC(t_tmp);
    auto secret_key = keygen.secret_key();
    t_sk = TOC(t_tmp); 

    PublicKey public_key;
    TIC(t_tmp);
    keygen.create_public_key(public_key);
    t_pk = TOC(t_tmp);

    RelinKeys relin_keys;
    TIC(t_tmp);
    keygen.create_relin_keys(relin_keys);
    t_rk = TOC(t_tmp);

    GaloisKeys gal_keys;
    TIC(t_tmp);
    keygen.create_galois_keys(gal_keys);
    t_gk = TOC(t_tmp);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    cout << "Public Key        : " << t_pk.count() <<endl;
    cout << "Secret Key Gen    : " << t_sk.count() <<endl;
    cout << "Rotation Key      : " << t_gk.count() <<endl;
    cout << "Relinearilzation  : " << t_rk.count() <<endl;

    // Prepare encoders.
    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "==== Number of slots: " << slot_count << endl;
    cout << "==== End of key generation & context creation ====" << endl;
    cout << "==================================================" << endl;

    if (argc > 1) num_batch = atoi(argv[1]);
    int N = batchSize*num_batch;
    std::vector<double> x(N);
#pragma omp parallel for
   for (int i=0; i<N; i++) x[i] = double(rand())/RAND_MAX - 0.5;
    //for (int i=0; i<N; i++) x[i] = double(1.0);

    // Encode & Encrypt.
    vector<Ciphertext> cipher_txts(num_batch);
    unsigned int batch_sz = slot_count;

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
    Plaintext  x_plain;
    Ciphertext x_encrypted;

    TIC(t_tmp);
    #pragma omp parallel for 
    for (int i=0; i< num_batch; i++){

        Plaintext  x_plain;
        Ciphertext x_encrypted;

        vector<double> x_batch = {x.begin() + i*batch_sz, x.begin() + (i+1)*batch_sz};

        encoder.encode(x_batch, scale, x_plain); 
        encryptor.encrypt(x_plain, x_encrypted);
        cipher_txts[i] = x_encrypted;
    }

    t_enc = TOC(t_tmp);
    cout << "End of encode/encrypt..." << endl;

    cout << "Compute x^2 and relinearize..." << endl;

    /// sum the whole slot with rotations.
    int num_rotation = log2(slot_count);

    TIC(t_tmp);
    evaluator.square_inplace(cipher_txts[0]);
    
    //evaluator.relinearize_inplace(cipher_txts[0], relin_keys);
    //evaluator.rescale_to_next_inplace(cipher_txts[0]);

    for (int i=1; i < num_batch; i++){

        evaluator.square_inplace(cipher_txts[i]);
        
        //evaluator.relinearize_inplace(cipher_txts[i], relin_keys);
        //evaluator.rescale_to_next_inplace(cipher_txts[i]);

        evaluator.add(cipher_txts[0], cipher_txts[i], cipher_txts[0]);
    }
    t_batch_sum = TOC(t_tmp);


    TIC(t_tmp);
    evaluator.relinearize_inplace(cipher_txts[0], relin_keys);
    evaluator.rescale_to_next_inplace(cipher_txts[0]);
    Ciphertext rotated;
    /// sum the batch by rotation.
    for (int j=1; j <= num_rotation; j++){
        int rot_steps = (int)pow(2.0, num_rotation-j);
        evaluator.rotate_vector(cipher_txts[0], rot_steps, gal_keys, rotated);
        evaluator.add(cipher_txts[0], rotated, cipher_txts[0]);
    } 
    t_reduct_sum = TOC(t_tmp);


    Plaintext plain_result;

    TIC(t_tmp);
    decryptor.decrypt(cipher_txts[0], plain_result); 
    encoder.decode(plain_result, result_vec);
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
    cout << "Public Key        : " << t_pk.count() <<endl;
    cout << "Secret Key Gen    : " << t_sk.count() <<endl;
    cout << "Rotation Key      : " << t_gk.count() <<endl;
    cout << "Relinearilzation  : " << t_rk.count() <<endl << endl;

    cout << "Prepare Input     : " << t_input.count() <<endl;
    cout << "Encode/Encrypt    : " << t_enc.count() <<endl;
    cout << "Decode/Decrypt    : " << t_dec.count() <<endl;
    cout << "Compute  per b    : " << t_batch_sum.count() / num_batch <<endl;
    cout << "EvalSum           : " << t_reduct_sum.count()  <<endl;
    
    cout << "Total: " << t_total.count()  << endl;

    cout << "[SEAL_Summary] " << num_batch  << " "
                         << t_pk.count() + t_sk.count() << " " << t_gk.count()<< " " << t_rk.count() << " "
                         << t_enc.count()   << " " << t_dec.count()<< " " << t_batch_sum.count()<< " " <<  t_reduct_sum.count() << " "
                         << t_total.count() <<  " " << t_raw.count() << " " << error  << " "
                         << batchSize << " "<<  sf << " " << FIRSTBIT << " " << rerr << endl;
    return 0;
}


