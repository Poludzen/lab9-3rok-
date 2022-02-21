#include "cstdio"
#include "ctime"

__global__ void GPU(const double* d, double* out){
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = 1; i < d[ind]; i++){
        out[ind] += i;
    }
}

// CPU example with showing time of execution
void testCPU(int count, const double* arr){
    clock_t start = clock();
    auto* out = new double[count];
    for(int j = 0; j < count; j++){
        for(int i = 1; i < arr[j]; i++){
            out[j] += i;
        }
    }
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds);
}

// GPU example with showing time of execution
void testGPU(int count, const double* arr){
    clock_t start = clock();
    double* aa, *out_;
    cudaMalloc(&aa, sizeof(double) * count);
    cudaMalloc(&out_, sizeof(double) * count);

    cudaMemcpy(aa, arr, sizeof(double) * count, cudaMemcpyHostToDevice);
    GPU<<<count / 100,100>>>(aa, out_);
    auto* out_c = (double*)malloc(sizeof(double) * count);
    cudaMemcpy(out_c, out_, sizeof(double ) * count, cudaMemcpyDeviceToHost);
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds);

    cudaFree(aa);
    cudaFree(out_);
}
int main()
{
    // starting experiment 
    int count = 1000000;
    auto* arr = new double[count];
    for(int i = 0; i < count; i++){
        arr[i] = 100000;
    }
    // testing
    //testCPU(count, arr);
    testGPU(count, arr);



    return 0;
}
