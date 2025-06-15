#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <chrono>
#include <random>
#include <cstdlib>

using namespace std;

__global__ void kernelAdd1d(float* a, float* b, float* result, int n) 
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result[i] = a[i] + b[i];
    }
}

__global__ void kernelSub1d(float* a, float* b, float* result, int n) 
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result[i] = a[i] - b[i];
    }
}

__global__ void kernelMul1d(float* a, float* b, float* result, int n) 
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result[i] = a[i] * b[i];
    }
}

__global__ void kernelDiv1d(float* a, float* b, float* result, int n) 
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result[i] = a[i] / b[i];
    }
}

static void test(int n, int numBlocks, int blockSize, int op, 
    float* h_a, float* h_b, float* h_result)
{
    float* d_a;
    float* d_b;
    float* d_result;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    switch (op) 
    {
        case 0:
            kernelAdd1d <<<numBlocks, blockSize>>> (d_a, d_b, d_result, n);
            break;
        case 1:
            kernelSub1d <<<numBlocks, blockSize>>> (d_a, d_b, d_result, n);
            break;
        case 2:
            kernelMul1d <<<numBlocks, blockSize>>> (d_a, d_b, d_result, n);
            break;
        case 3:
            kernelDiv1d <<<numBlocks, blockSize>>> (d_a, d_b, d_result, n);
            break;
        defult:
            printf("Ошибка в выборе оперции!!!");
            return;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

int main(int argc, char* argv[]) 
{
    int rounds = atoi(argv[1]);

    int n = atoi(argv[2]);

    int numBlocks = atoi(argv[3]);
    int blockSize = atoi(argv[4]);

    double time = 0;

    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_result = new float[n];

    mt19937 gen(42);
    uniform_real_distribution<float> dist(1, 1000);

    for (int round = 0; round < rounds; round++)
    {
        for (int i = 0; i < n; i++)
        {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
        }

        auto start = chrono::high_resolution_clock::now();

        test(n, numBlocks, blockSize, 0, h_a, h_b, h_result);
        test(n, numBlocks, blockSize, 1, h_a, h_b, h_result);
        test(n, numBlocks, blockSize, 2, h_a, h_b, h_result);
        test(n, numBlocks, blockSize, 3, h_a, h_b, h_result);
        cout << h_a[0] << " " << h_a[0] << " " << h_result[0] << endl;

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        time += elapsed.count();
        cout << elapsed.count() << endl;

    }
    time /= rounds;
    cout << "time = " << time << " s" << endl;

    return 0;
}