#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include "CSVRow.h"
#include "preprocessing.h"

#define N_VARIABLES 4
#define N_DATA 100000
#define ITERATIONS 3000
#define LR 0.1
#define MAX_THREADS 1024

__global__ void averageBins(float *d_leafBins, float *d_residual, float *d_leafValue, int max) {
    int i = threadIdx.x;

    for (int j = 0; j < max, j++) {
        d_leafValue[i] += d_residual[d_leafBins[i]];
    }

    d_leafValue[i] /= float(d_leafBins[i].size());
}

__global__ void getNewPredictions(float *d_predicted, float *d_leafValue, float *d_leafAssignment, float lr) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d_predicted[i] += lr * d_leafValue[d_leafAssignment[i]];
}

__global__ void getNewResiduals(float *d_actual, float *d_predicted, float *d_residual) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d_residual[i] = d_actual[i] - d_predicted[i];
}

void averageBins(float &leafBins, float &residual, float &leafValue) {
    for (int i = 0; i < leafBins.size(), i++) {
        for (int j = 0; j < leafBins[i].size(), j++) {
            leafValue[i] += residual[leafBins[i][j]];
        }

        leafValue[i] /= leafBins[i].size();
    }

}

void getNewPredictions(float &predicted, float &leafValue, float &leafAssignment, float lr) {
    for (int i = 0; i < predicted.size(); i++) {
        predicted[i] += lr * leafValue[leafAssignment[i]];
    }
}

void getNewResiduals(float &actual, float &predicted, float &residual) {
    for (int i = 0; i < actual.size(); i++) {
        residual[i] = actual[i] - predicted[i];
    }
}

int main()
{
	std::string filename = std::to_string(N_VARIABLES) + "d.txt";
    std::ifstream       file(filename);

    CSVRow				variable;

    // data_table[i][j] corresponds to the ith data point and jth variable. If j = N_VARIABLES, j
    // is the output of the ith data point
    CSVRow              data_table[N_VARIABLES + 1];
    int row = 0;
    while(file >> variable)
    {
        data_table[row] = variable;
    	std::cout << "4th Element(" << data_table[row][3] << ")\n";
        row++;
    }


    // initialize data
    float leafBins[N_VARIABLES] __attribute__((aligned(64)));
    float leafAssignment[data_table.size()] __attribute__((aligned(64)));
    float residual[data_table.size()] __attribute__((aligned(64)));
    float leafValue[data_table.size()] __attribute__((aligned(64)));
    float actual[data_table.size()] __attribute__((aligned(64)));
    float predicted[data_table.size()] __attribute__((aligned(64)));
    float resultGPU[data_table.size()] __attribute__((aligned(64)));
    memcpy(actual, data_table[data_table.size()-1], data_table.size()*sizeof(float));

    // fill arrays
    preprocessing(actual, predicted);
    std::fill_n(leafBins, data_table.size(), 0);
    std::fill_n(leafAssignment, data_table.size(), 0)
    std::fill_n(residual, data_table.size(), 0);
    std::fill_n(leafValue, data_table.size(), 0);

    // compute on CPU
    std::cout << "Begin CPU calculations" << std::endl;
    begin_roi();
    leafAssign(leafBins, leafAssignment);
    for (int i = 0; i < ITERATIONS; i++) {
        getNewResiduals(actual, predicted, residual);
        averageBins(leafBins, residual, leafValue);
        getNewPredictions(predicted, leafValue, leafAssignment, LR);
    }
    end_roi();

    // Allocate memory
    cudaError_t err = cudaSuccess;
    size_t size_input = data_table.size() * N_VARIABLES * sizeof(float);
    size_t size_output = data_table.size() * sizeof(float);

    float *d_leafBins;
    float *d_leafAssignment;
    float *d_actual;
    float *d_predicted;
    float *d_residual;
    float *d_leafValue;

    // allocate d_leafBins memory
    err = cudaMalloc((void **)&d_leafBins, size_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_leafBins (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // allocate d_leafAssignment memory
    err = cudaMalloc((void **)&d_leafAssignment, size_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_leafAssignment (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // allocate d_actual memory
    err = cudaMalloc((void **)&d_actual, size_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_actual (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // allocate d_predicted memory
    err = cudaMalloc((void **)&d_predicted, size_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_predicted (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // allocate d_residual memory
    err = cudaMalloc((void **)&d_residual, size_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_residual (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // allocate d_leafValue memory
    err = cudaMalloc((void **)&d_leafValue, size_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_leafValue (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // transfer memory associated with initalized vars from host to device
    err = cudaMemcpy(d_leafBins, leafBins, size_output, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_leafAssignment,leafAssignment, size_output, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_actual, actual, size_output, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_predicted, predicted, size_output, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_residual, residual, size_output, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_leafValue, leafValue, size_output, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //compute numBlocks, numThreads
    int numBlocks = data_table.size() / MAX_THREADS, MAX_THREADS;
    int numThreads = pow(2, N_VARIABLES);

    //compute on GPU
    std::cout << "Begin GPU calculations" << std::endl;
    begin_roi();
    leafAssign<<<>>>(d_leafBins, d_leafAssignment);
    cudaDeviceSynchronize();
    for (int i = 0; i < ITERATIONS; i++) {
        getNewResiduals<<<numBlocks, MAX_THREADS>>>(d_actual, d_predicted, d_residual);
        cudaDeviceSynchronize();
        averageBins<<<1, numThreads>>>(d_leafBins, d_residual, d_leafValue);
        cudaDeviceSynchronize();
        getNewPredictions<<<numBlocks, MAX_THREADS>>>(d_predicted, d_leafValue, d_leafAssignment, LR);
        cudaDeviceSynchronize();
    }
    end_roi();


    // transfer memory from device to host
    err = cudaMemcpy(resultGPU, d_predicted, size_output, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_predicted from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    compare(predicted, resultGPU, data_table.size());


    // Free device memory
    err = cudaFree(d_leafBins);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_leafBins (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_leafAssignment);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_leafBins (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_predicted);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_predicted (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_actual);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_actual (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_residual);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_residual (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_leafValue);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_leafValue (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}