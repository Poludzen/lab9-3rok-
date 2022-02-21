# lab9-3rok-
## In this paper we want to compare: is GPU(CUDA) really faster than CPU on the same operations on vectors(code was given)
### First: installation CUDA on GoogleColab 
To install cuda we need to run this codes:
```
!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove
!apt-get update
```
Then:
```
!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
!apt-get update
!apt-get install cuda-9.2
```
Next check installed version:
```
!nvcc --version
```
load plugin
```
%load_ext nvcc_plugin
```
Now we are done! To run code with CUDA we need to paste  ```%%cu``` in the start of the cell 

## First example: increasing values at vectors
### What lengths of vectors were in test?
This lengths were choosen : 10,50,100,200,500,1000,5000,10000,50000,100000,200000,500000,1000000

### Code(with all counts(sizes) on CPU and GPU):
```c++
%%cu
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
    int counts[] = {10,50,100,200,500,1000,5000,10000,50000,100000,200000,500000,1000000};
    for(int j=0;j<13;j++){
      int count = counts[j];
      auto* arr = new double[count];
      for(int i = 0; i < count; i++){
          arr[i] = 100000;
      }
      // testing
      testCPU(count, arr);
      //testGPU(count, arr);
    }
    return 0;
}
```

### Result of calculations:
##### Table of results
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_table.jpg?raw=True "Results")
#### Diagrams of results:
##### On counts : 10,50,100,200
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_time-1.jpg?raw=True "Diagram 1")
##### On counts : 10,50,100,200,500,1000
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_time-4.jpg?raw=True "Diagram 2")
##### On all counts:
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_time-3.jpg?raw=True "Diagram 3")

### Conclusion:
The use of GPU(CUDA) is effective for large vector sizes if we compare with CPU (in our example, starting from the size of 200, this can be seen in diagrams 2 and 3). On small sizes, due to additional initialization operations, the use of GPU(CUDA) is inefficient, as can be easily seen in the first diagram. Moreover, with a size of 200, the difference is insignificant, but with a size of 1.000.000. the GPU(CUDA) is 125 times more efficient than CPU.

## Second example : Single-precision A*X Plus Y(SAXPY)
We will do same comparing, but on another sizes and on another algorithm
### Sizes that were in the test:
500000,1000000,2500000,5000000,10000000,50000000,100000000
### Code for GPU(with all counts(sizes)):
```c++
%%cu
#include <stdio.h>
#include "cstdio"
#include "ctime"

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int counts[] = { 500000,1000000,2500000,5000000,10000000,50000000,100000000 };
  for(int j = 0;j<7;j++){
    int N = counts[j];
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    clock_t start = clock();
    // Perform SAXPY on N elements
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = max(maxError, abs(y[i]-4.0f));
    
    //printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
  }
}
```
### Code for CPU
```c++
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
// same exampel but on cpu
// let's not calculate loss
// saxpy
void saxpy_cpu(float* x, float* y, int N, float a) {
    for (int i = 0; i < N; i++) y[i] = a * x[i] + y[i];
}
// main
int main() {
    // sizes of arrays
    int counts[] = { 500000,1000000,2500000,5000000,10000000,50000000,100000000};
    for (int j = 0; j < 7; j++) {
        int N = counts[j];
        float* x, * y;
        x = (float*)malloc(N * sizeof(float));
        y = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
        // experiment
        clock_t start = clock();
        saxpy_cpu(x, y, N, 2.0f);
        clock_t end = clock();
        double seconds = (double)(end - start) / CLOCKS_PER_SEC;
        printf("The time: %f seconds\n", seconds);
        free(x);
        free(y);
    }
    return 0;
}
```
### Result of calculations:
##### Table of results
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu2_table.jpg?raw=True "Table 2")
#### Diagrams of results:
##### On counts :  500000,1000000
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu2_time1.jpg?raw=True "Diagram2-1")
##### On all counts:
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu2_time2.jpg?raw=True "Diagram2-2")
### Conclusion:
Again: usage of GPU(CUDA) is so effective with large vectors. In this example we take only large vectors, and it is easy to see that GPU works much faster.
