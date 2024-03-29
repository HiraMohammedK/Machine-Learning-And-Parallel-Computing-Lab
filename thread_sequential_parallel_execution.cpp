#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>

using namespace std;


void generateRandomArray(int arr[], int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 1000; 
    }
}


int sequentialSum(const int arr[], int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum;
}

bool sequentialSearch(const int arr[], int size, int key) {
    for (int i = 0; i < size; ++i) {
        if (arr[i] == key) {
            return true;
        }
    }
    return false;
}

void parallelComputation(const int arr[], int size, int& sum, int key, bool& found, int start, int end) {
    for (int i = start; i < end; ++i) {
        sum += arr[i];
        if (arr[i] == key) {
            found = true;
        }
    }
}

int main() {
    srand(time(0));

    vector<int> sizes = {100, 500, 1000, 2000, 5000, 100000};

    for (int size : sizes) {
        int* arr = new int[size];
        generateRandomArray(arr, size);


        auto startSeq = chrono::high_resolution_clock::now();
        int seqSum = sequentialSum(arr, size);
        bool seqSearch = sequentialSearch(arr, size, rand() % 1000); 
        auto endSeq = chrono::high_resolution_clock::now();
        chrono::duration<double> seqDuration = endSeq - startSeq;

        int numThreads = 1;
        int chunkSize = size / numThreads;

        auto startPar = chrono::high_resolution_clock::now();
        vector<thread> threads;
        int parSum = 0;
        bool parSearch = false;

        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(parallelComputation, arr, size, ref(parSum), rand() % 1000, ref(parSearch), i * chunkSize, (i + 1) * chunkSize);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto endPar = chrono::high_resolution_clock::now();
        chrono::duration<double> parDuration = endPar - startPar;

        cout << "Array Size: " << size << endl;
        cout << "Sequential Sum: " << seqSum << ", Search Result: " << (seqSearch ? "Found" : "Not Found") << endl;
        cout << "Sequential Execution Time: " << seqDuration.count() << " seconds" << endl;
        cout << "Parallel Sum: " << parSum << ", Search Result: " << (parSearch ? "Found" : "Not Found") << endl;
        cout << "Parallel Execution Time: " << parDuration.count() << " seconds" << endl;

        delete[] arr;
    }

    return 0;
}
