#include "hip/hip_runtime.h"
#include <stdio.h>
#include <hip/hip_runtime.h>

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    hipError_t err = hipSuccess;

    int numElements = 1<<16; //16384

    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
    
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    hipMalloc((void **)&d_A, size);
    hipMalloc((void **)&d_B, size);
    hipMalloc((void **)&d_C, size);

    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(2*numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0>>>(d_A, d_B, d_C, 2*numElements);

    // hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
