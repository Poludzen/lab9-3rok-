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
    int counts[] = { 500000,1000000,2500000,5000000,10000000,50000000,100000000 };
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
