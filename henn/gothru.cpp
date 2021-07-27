#include <iostream>
using namespace std;

#include "hdf5.h"
#include <cstdio>

#include "H5Cpp.h"
using namespace H5;

herr_t op_func (hid_t loc_id, const char *name, const H5O_info_t *info,
            void *operator_data);

herr_t op_func_L (hid_t loc_id, const char *name, const H5L_info_t *info,
            void *operator_data);

int main (int argc, char* argv[])
{
    herr_t          status;
    hid_t           file;           /* Handle */
  
    file = H5Fopen (argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
                          
    printf ("Objects in the file:\n");
    status = H5Ovisit (file, H5_INDEX_NAME, H5_ITER_NATIVE, op_func, NULL);

    status = H5Fclose (file);

    return 0;
}


herr_t op_func (hid_t loc_id, const char *name, const H5O_info_t *info,
            void *operator_data)
{
    printf ("/"); 
    if (name[0] == '.')         /* Root group, do not print '.' */
        printf ("  (Group)\n");
    else
        switch (info->type) {
            case H5O_TYPE_GROUP:
                printf ("%s  (Group)\n", name);
                break;
            case H5O_TYPE_DATASET:
                printf ("%s  (Dataset)\n", name);
                
                break;
            case H5O_TYPE_NAMED_DATATYPE:
                printf ("%s  (Datatype)\n", name);
                break;
            default:
                printf ("%s  (Unknown)\n", name);
        }

    return 0;
}

