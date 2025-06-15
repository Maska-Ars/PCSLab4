#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono>
#include <random>
#include <cstdlib>

using namespace std;

__global__ void sumKernel(const float* data, float* sum_ptr, int size) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) 
    {
        __shared__ float shared_sum[1024];
        shared_sum[threadIdx.x] = data[index];
        __syncthreads();

        for (int i = 1; i < blockDim.x; i <<= 1) 
        {
            if (threadIdx.x < i) 
            {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) 
        {
            atomicAdd(sum_ptr, shared_sum[0]);
        }
    }
}

static void test(int n, int numBlocks, int blockSize, float* h_a, float& h_sum)
{
    float* d_sum;
    float* d_a;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    sumKernel << <numBlocks, blockSize >> > (d_a, d_sum, n);

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Sum = " << h_sum << endl;

    cudaFree(d_a);
    cudaFree(d_sum);
}

int main(int argc, char* argv[])
{
    int rounds = atoi(argv[1]);
    int n = atoi(argv[2]);
    int blockSize = atoi(argv[3]);

    float h_sum = 0.0f;
    double time = 0;

    float* h_a = new float[n];

    int numBlocks = (n + blockSize - 1) / blockSize;

    mt19937 gen(42);
    uniform_real_distribution<float> dist(1, 1000);

    for (int round = 0; round < rounds; round++)
    {

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < n; i++)
        {
            h_a[i] = dist(gen);
        }
        test(n, numBlocks, blockSize, h_a, h_sum);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        time += elapsed.count();
        cout << endl;
    }

    time /= rounds;
    cout << "time = " << time << " s" << endl;

    return 0;
}