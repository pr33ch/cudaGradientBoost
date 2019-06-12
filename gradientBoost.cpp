#include <iostream>
#include <string>
#include <algorithm>
#include "CSVRow.h"
#include "preprocessing.h"

#define N_VARIABLES 4
#define N_DATA 100000

__global__ void averageBins(float *d_leafAssignment, float *d_residual, float *d_leafValue, int max) {
    int i = threadIdx.x;

    for (int j = 0; j < max, j++) {
        d_leafValue[i] += d_residual[d_leafAssignment[i]]
    }

    d_leafValue[i] /= float(d_leafAssignment[i].size())
}

__global__ void getNewPredictions(float *d_predicted, float *d_leafValue, float lr) {
    int i = blockDim.x * blockIdx.x + threadIdx.x

    d_predicted[i] += lr * d_leafValue[i]
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
    float leafAssignment[data_table.size()] __attribute__((aligned(64)));
    float residual[data_table.size()] __attribute__((aligned(64)));
    float leafValue[data_table.size()] __attribute__((aligned(64)));
    float actual[data_table.size()] __attribute__((aligned(64)));
    float predicted[data_table.size()] __attribute__((aligned(64)));
    memcpy(actual, data_table[data_table.size()-1], data_table.size()*sizeof(float));

    // fill arrays
    preprocessing(actual, predicted);
    std::fill_n(leafAssignment, data_table.size(), 0);
    std::fill_n(residual, data_table.size(), 0);
    std::fill_n(leafValue, data_table.size(), 0);

    // Allocate memory
    cudaError_t err = cudaSuccess;
    size_t size_input = data_table.size() * N_VARIABLES * sizeof(float);
    size_t size_output = data_table.size() * sizeof(float);

    float *d_leafAssignment;
    float *d_actual;
    float *d_predicted;
    float *d_residual;
    float *d_leafValue;

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
    err = cudaMemcpy(d_leafAssignment, leafAssignment, size_output, cudaMemcpyHostToDevice);
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

    // Free device memory
    err = cudaFree(d_leafAssignment);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_leafAssignment (error code %s)!\n", cudaGetErrorString(err));
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