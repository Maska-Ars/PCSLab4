#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <chrono>
#include <random>
#include <cstdlib>

using namespace std;

__global__ void kernelAdd2d(float* a, float* b, float* result, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result[i] = a[i] + b[i];
    }
}

__global__ void kernelSub2d(float* a, float* b, float* result, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result[i] = a[i] - b[i];
    }
}

__global__ void kernelMul2d(float* a, float* b, float* result, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result[i] = a[i] * b[i];
    }
}

__global__ void kernelDiv2d(float* a, float* b, float* result, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result[i] = a[i] / b[i];
    }
}

static void test(int m, int n, int numBlocks, int blockSize, int op,
    float** h_a, float** h_b, float** h_result)
{
    float* d_a;
    float* d_b;
    float* d_result;

    float* h_a_1d = new float[m * n];
    float* h_b_1d = new float[m * n];
    float* h_result_1d = new float[m * n];

    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            h_a_1d[n * i + j] = h_a[i][j];
            h_b_1d[n * i + j] = h_a[i][j];
        }
    }

    cudaMalloc(&d_a, m *n * sizeof(float));
    cudaMalloc(&d_b, m * n * sizeof(float));
    cudaMalloc(&d_result, m * n * sizeof(float));

    cudaMemcpy(d_a, h_a_1d, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_1d, m * n * sizeof(float), cudaMemcpyHostToDevice);

    switch (op)
    {
        case 0:
            kernelAdd2d << <numBlocks, blockSize >> > (d_a, d_b, d_result, m * n);
            break;
        case 1:
            kernelSub2d << <numBlocks, blockSize >> > (d_a, d_b, d_result, m * n);
            break;
        case 2:
            kernelMul2d << <numBlocks, blockSize >> > (d_a, d_b, d_result, m * n);
            break;
        case 3:
            kernelDiv2d << <numBlocks, blockSize >> > (d_a, d_b, d_result, m * n);
            break;
        defult:
            printf("Ошибка в выборе оперции!!!");
            return;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_result_1d, d_result, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            h_result[i][j] = h_result_1d[n * i + j];
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

int main(int argc, char* argv[])
{
    int rounds = atoi(argv[1]);

    const int m = atoi(argv[2]);
    const int n = atoi(argv[3]);

    int numBlocks = atoi(argv[4]);
    int blockSize = atoi(argv[5]);

    double time = 0;

    float** h_a = new float* [m];
    float** h_b = new float* [m];
    float** h_result = new float* [m];

    for (int i = 0; i < m; i++)
    {
        h_a[i] = new float[n];
        h_b[i] = new float[n];
        h_result[i] = new float[n];

    }

    mt19937 gen(42);
    uniform_real_distribution<float> dist(1, 1000);

    for (int round = 0; round < rounds; round++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                h_a[i][j] = dist(gen);
                h_b[i][j] = dist(gen);
            }
        }

        auto start = chrono::high_resolution_clock::now();

        test(m, n, numBlocks, blockSize, 0, h_a, h_b, h_result);
        test(m, n, numBlocks, blockSize, 1, h_a, h_b, h_result);
        test(m, n, numBlocks, blockSize, 2, h_a, h_b, h_result);
        test(m, n, numBlocks, blockSize, 3, h_a, h_b, h_result);
        cout << h_a[0][0] << " " << h_b[0][0] << " " << h_result[0][0] << endl;

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        time += elapsed.count();
        cout << elapsed.count() << endl;

    }
    time /= rounds;
    cout << "time = " << time << " s" << endl;

    return 0;
}