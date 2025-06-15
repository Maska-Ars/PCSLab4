#include <iostream>
#include <chrono>
#include <random>
#include <cstdlib>

using namespace std;

void merge(float arr[], int left, int mid, int right) 
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    float* L = new float[n1];
    float* R = new float[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(float arr[], int left, int right) 
{
    if (left < right) 
    {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main(int argc, char* argv[])
{
    int rounds = atoi(argv[1]);
    int n = atoi(argv[2]);

    float* arr = new float[n];

    double time = 0;

    mt19937 gen(42);
    uniform_real_distribution<float> dist(1, 100000);

    for (int round = 0; round < rounds; round++)
    {

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < n; i++)
        {
            arr[i] = dist(gen);
        }

        cout << arr[0] << " " << arr[1] << " " << arr[2] << endl;

        mergeSort(arr, 0, n - 1);

        cout << arr[0] << " " << arr[1] << " " << arr[2] << endl;

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        time += elapsed.count();
        cout << elapsed.count() << endl;
    }

    time /= rounds;
    cout << "time = " << time << " s" << endl;

    return 0;
}