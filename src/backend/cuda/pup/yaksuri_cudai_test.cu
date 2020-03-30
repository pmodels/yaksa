#include <cuda.h>
#include "yaksuri_cudai_pup.h"

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int yaksuri_cudai_test(void)
{
    vector_add<<<1,1>>>(NULL, NULL, NULL, 0);

    return 0;
}
