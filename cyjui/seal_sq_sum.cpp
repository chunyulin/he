#include<fstream>
#include "seal/seal.h"
//#include "examples.h"

#include <iostream>
#include <omp.h>

/// For performance profiling.
#include <chrono>
typedef  std::chrono::duration<double> DURATION;
#undef TIC
#undef TOC
#define TIC(t) t = std::chrono::system_clock::now();
#define TOC(t) (std::chrono::system_clock::now() - t);

std::chrono::time_point<std::chrono::system_clock> t_tmp, t_global;

DURATION t_pk, t_sk, t_rk, t_gk;        /// key related.
DURATION t_input, t_enc, t_dec, t_square, t_batch_sum, t_reduct_sum;   /// computation related.
DURATION t_raw, t_total;

using namespace std;
using namespace seal;

int main(int argc, char* argv[])
{
    // Get the num of cores.
    int num_cores   = omp_get_max_threads();
    int num_threads = omp_get_num_threads();
    string input_bin = "1M.bin";
    string dst_base  = "";
    
    cout << "==== # of cores   : " << num_cores   << " ==== " << endl;
    cout << "==== # of threads : " << num_threads << " ==== " << endl;
    
    if (argc >= 3) {
        num_threads = atoi(argv[1]);
        input_bin   = argv[2];
        dst_base    = argv[3];
    }
    TIC(t_global);
        
    /// Set openMP threads.
    //omp_set_num_threads(num_threads);
    omp_set_num_threads(num_cores);
    
    // Encryption parameters. 
    EncryptionParameters parms(scheme_type::ckks);

    // Set prms for polynomial ring.
    int batchSize=8192;
    //int batchSize=4096;
    size_t poly_modulus_degree = batchSize*2;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
    //parms.set_coeff_modulus( { 0x7fffffd8001, 0x7fffffc8001, 0xfffffffc001, 0xffffff6c001, 0xfffffebc001 } );
    //parms.set_coeff_modulus( { 0xfffffffd8001, 0xfffffffa0001, 0xfffffff00001, 0x1fffffff68001, 0x1fffffff50001, 
    //                                                                    0x1ffffffee8001, 0x1ffffffea0001, 0x1ffffffe88001, 0x1ffffffe48001 } );

   //auto p128 = seal::util::global_variables::GetDefaultCoeffModulus128();
   
    double scale = pow(2.0, 39);

    // Print the context.
    SEALContext context(parms);
    //print_parameters(context);
    cout << endl;

    cout << " ==== Elapsed time for creating SEALContext ==== " << endl;
    cout << "=================================================" << endl;

    // Here we start key generation.

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
   
    /// Load the binary file as input.
    //TIC(t_tmp);
    //vector<double> input = read_bin(input_bin);
    //unsigned long N = input.size();
    //t_input = TOC(t_tmp);

    int num_batch = 10;
    int N = batchSize*num_batch;
    std::vector<double> x(N);
    #pragma omp parallel for
    for (int i=0; i<N; i++) x[i] = (double) (rand()-RAND_MAX/2) / (RAND_MAX);

    
    //cout << "Input vector: " << endl;
    //print_vector(input, 3, 7);

    //int num_batch = N/slot_count + int(N%slot_count!=0);
    
    // print_line(__LINE__);
    
    // Encode & Encrypt.
    vector<Ciphertext> cipher_txts(num_batch);
    unsigned int batch_sz = slot_count;

    cout << "# of data  : " << N << endl; 
    cout << "num_batch  : " << num_batch << endl;
    cout << "slot_count : " << slot_count << endl;
    
    TIC(t_tmp);
    double raw_result = 0.0;
    
    /// calculate the raw result.
    #pragma omp parallel for reduction(+ : raw_result)
    for(int i=0; i < x.size(); i++){
#ifdef SQUARE
        raw_result += x[i]*x[i];
#else
        raw_result += x[i];
#endif
    }
    t_raw = TOC(t_tmp);
    
    /// Encode & Encrypt.
    Plaintext  x_plain;
    Ciphertext x_encrypted;
    
    TIC(t_tmp);
    #pragma omp parallel for 
    for (int i=0; i< num_batch-1; i++){

        Plaintext  x_plain;
        Ciphertext x_encrypted;
        
        vector<double> x_batch = {x.begin() + i*batch_sz, x.begin() + (i+1)*batch_sz};
        /// encode & encrypt.
        encoder.encode(x_batch, scale, x_plain); 
        encryptor.encrypt(x_plain, x_encrypted);
        cipher_txts[i] = x_encrypted;
    }
    
    /// deal with the last batch.
    vector<double> x_batch = {x.begin() + (num_batch-1)*batch_sz, x.end()};
    encoder.encode(x_batch, scale, x_plain);    
    encryptor.encrypt(x_plain, x_encrypted);
    cipher_txts[num_batch-1] = x_encrypted;

    t_enc = TOC(t_tmp);
//     cout << "Encode/Encrypt : " << t_enc.count() << endl;
    cout << "End of encode/encrypt..." << endl;

    /// do the HE computations.
    /// calculate the square.
    cout << "Compute x^2 and relinearize..." << endl;
    
    
    /// sum the whole slot with rotations.
    int num_rotation = log2(slot_count);
    vector<double> result_vec(slot_count);
    double he_result = 0.0;
    
    TIC(t_tmp);
    #pragma omp parallel for
    for (int i=0; i < num_batch; i++){
        
#ifdef SQUARE
        evaluator.square_inplace(cipher_txts[i]);
        evaluator.relinearize_inplace(cipher_txts[i], relin_keys);
        evaluator.rescale_to_next_inplace(cipher_txts[i]);
#endif
//         Ciphertext rotated_sum = cipher_txts[i];
        Ciphertext rotated;
        
        /// sum the batch by rotation.
        for (int j=1; j <= num_rotation; j++){
            int rot_steps = (int)pow(2.0, num_rotation-j);
            evaluator.rotate_vector(cipher_txts[i], rot_steps, gal_keys, rotated);
            evaluator.add(cipher_txts[i], rotated, cipher_txts[i]);
        } 
//         cipher_txts[i] = rotated_sum;
    }
    t_batch_sum = TOC(t_tmp);
    
    TIC(t_tmp);
   
    
    Plaintext plain_result;
    Ciphertext cipher_result = cipher_txts[0];
    
    TIC(t_tmp);
    for (int i=1; i < num_batch; i++){
        evaluator.add(cipher_result, cipher_txts[i], cipher_result);
    }
    t_reduct_sum = TOC(t_tmp);
    
    
    TIC(t_tmp);
    decryptor.decrypt(cipher_result, plain_result); 
    encoder.decode(plain_result, result_vec);
    he_result = result_vec[0];
    t_dec = TOC(t_tmp);
    
    /// time for whole calculation.
    t_total = TOC(t_global);
    
    cout << "Decrypted   : " << t_dec.count() << endl;
    printf( "CKKS Result : %.6f\n", he_result);
    printf( "Raw  Result : %.6f\n", raw_result);
    
    
    /************************************
        Summarize the results.
    *************************************/
    double error = he_result - raw_result;
    
    vector<double> out_vec(20);
    
    /// statistics for key generation.
    out_vec[0] = t_pk.count();
    out_vec[1] = t_sk.count();
    out_vec[2] = t_gk.count();
    out_vec[3] = t_rk.count(); 
    
    /// statistics for computations.
    out_vec[4] = t_input.count();
    out_vec[5] = t_enc.count();
    out_vec[6] = t_dec.count();
    out_vec[7] = t_square.count(); 
    out_vec[8] = t_batch_sum.count(); 
    out_vec[9] = t_reduct_sum.count();
    out_vec[10] = t_raw.count();
    out_vec[11] = t_total.count();
    out_vec[12] = raw_result;
    out_vec[13] = he_result;
    out_vec[14] = num_batch;
    out_vec[15] = 0; // correspond to SumKey in palisade.
    
    cout << "======== Performance    ========" << endl;
    cout << "======== Key Generation ========" << endl;
    cout << "Public Key        : " << t_pk.count() <<endl;
    cout << "Secret Key Gen    : " << t_sk.count() <<endl;
    cout << "Rotation Key      : " << t_gk.count() <<endl;
    cout << "Relinearilzation  : " << t_rk.count() <<endl;
    
    cout << "======== Computations   ========" << endl;
    cout << "Prepare Input     : " << t_input.count() <<endl;
    cout << "Encode/Encrypt    : " << t_enc.count() <<endl;
    cout << "Decode/Decrypt    : " << t_dec.count() <<endl;
    cout << "Enc. Squaring     : " << t_square.count() <<endl;
    cout << "Enc. Sum [batch]  : " << t_batch_sum.count() / num_batch <<endl;
    cout << "Enc. Sum [reduce] : " << t_reduct_sum.count() <<endl;
    
    
    return 0;
}


