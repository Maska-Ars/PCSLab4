#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include<iostream>
#include <chrono>
#include <random>
#include <cstdlib>

using namespace std;

__global__ void bitonic_sort_step(float* d_m, int j, int k)
{
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    if ((ixj) > i) 
    {
        if ((i & k) == 0) 
        {
            if (d_m[i] > d_m[ixj]) 
            {
                float temp = d_m[i];
                d_m[i] = d_m[ixj];
                d_m[ixj] = temp;
            }
        }
        if ((i & k) != 0) 
        {
            if (d_m[i] < d_m[ixj]) 
            {
                float temp = d_m[i];
                d_m[i] = d_m[ixj];
                d_m[ixj] = temp;
            }
        }
    }
}

void bitonic_sort(int n, float* m, int numBlocks, int blockSize)
{
    float* d_m;

    cudaMalloc(&d_m, n * sizeof(float));
    cudaMemcpy(d_m, m, n, cudaMemcpyHostToDevice);

    for (int k = 2; k <= n; k <<= 1) 
    {
        for (int j = k >> 1; j > 0; j = j >> 1) 
        {
            bitonic_sort_step<<<numBlocks, blockSize>>>(d_m, j, k);
        }
    }

    cudaMemcpy(m, d_m, n, cudaMemcpyDeviceToHost);
    cudaFree(d_m);
}

int main(int argc, char* argv[])
{
    int rounds = atoi(argv[1]);
    int n = atoi(argv[2]);

    int numBlocks = atoi(argv[3]);
    int blockSize = atoi(argv[4]);

    double time = 0;

    float* m = new float[n];

    mt19937 gen(42);
    uniform_real_distribution<float> dist(0, 2^16);

    for (int round = 0; round < rounds; round++)
    {
        for (int i = 0; i < n; i++)
        {
            m[i] = dist(gen);
        }

        auto start = chrono::high_resolution_clock::now();
        bitonic_sort(n, m, numBlocks, blockSize);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        time += elapsed.count();
        cout << elapsed.count() << endl;
    }

    time /= rounds;
    cout << "time = " << time << " s" << endl;

    return 0;
}