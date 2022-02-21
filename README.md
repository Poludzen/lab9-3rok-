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

### What length of vectors was in test?
This lengths were choosen : 10,50,100,200,500,1000,5000,10000,50000,100000,200000,500000,1000000

### Code(with all counts):
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
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_time-2.jpg?raw=True "Diagram 2")
##### On all counts:
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_time-3.jpg?raw=True "Diagram 3")

## Conclusion:
The use of GPU(CUDA) is effective for large vector sizes if we compare with CPU (in our example, starting from the size of 200, this can be seen in diagrams 2 and 3). On small sizes, due to additional initialization operations, the use of GPU(CUDA) is inefficient, as can be easily seen in the first diagram. Moreover, with a size of 200, the difference is insignificant, but with a size of 1.000.000. the GPU(CUDA) is 125 times more efficient than CPU.

