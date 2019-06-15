// #include <algorithm>
#include "CSVRow.h"
#include <inttypes.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cooperative_groups.h>

#define N_VARIABLES 3
#define N_DATA 1024
#define ITERATIONS 3000
#define LR 0.1
#define MAX_THREADS 1024

void compare(float* cpu, float* gpu, int size) {
  bool error = false;
  for(int i = 0; i < size; ++i) {
      float diff = cpu[i] - gpu[i];
      if(diff > 3 || diff < -3) {
      error = true;
      break;
    }
  }
  if(error) {
    for(int i = 0; i < size; ++i) {
      std::cout << i << " " << cpu[i] << ":" << gpu[i];;

      float diff = cpu[i] - gpu[i];
      if(diff>3 || diff <-3) {
        std::cout << " \t\tERROR";
      }
      std::cout << "\n";
    }
  } else {
    std::cout << "results match\n";
  }
}

void initialize_tree(CSVRow *data_table, float * tree)
{
    for (int nth_variable = 0; nth_variable < N_VARIABLES; nth_variable++)
    {
        float average = 0;
        for (int nth_sample = 0; nth_sample < N_DATA; nth_sample ++)
        {
            average += data_table[nth_sample][nth_variable];
        }
        // tree[nth_variable] = average;
        average = average/N_DATA;
        memcpy(&tree[nth_variable], &average, sizeof(float));
    }
}

// naive CPU implementation of leaf assignment of each datapoint

// the structure of tree is a balanced binary decision tree, where the decisions at each depth i are
// whether or not a given data point's value for variable i is <= the average for variable i. The
// depth of the tree is the dimensionality of our dataset

void leaf_assign(CSVRow data_table[], float *tree, std::vector<int> *leaf_bins, int * leafAssignment)
{
    for (int nth_sample = 0; nth_sample < N_DATA; nth_sample ++)
    {
        int upper = pow(2, N_VARIABLES) - 1;
        int lower = 0;
        // perform binary search to classify sample
        for(int nth_variable = 0; nth_variable < N_VARIABLES; nth_variable ++)
        {
            if (nth_variable == N_VARIABLES - 1) // if we've reached the last decision node
            {
                if (data_table[nth_sample][nth_variable] <= tree[nth_variable])
                {
                    leafAssignment[nth_sample] = lower;
                    leaf_bins[lower].push_back(nth_sample);
                }
                else
                {
                    leafAssignment[nth_sample] = upper;
                    leaf_bins[upper].push_back(nth_sample);
                }
            }
            if (data_table[nth_sample][nth_variable] <= tree[nth_variable])
            {
                upper = upper/2;
            }
            else
            {
                lower = (upper-lower)/2 + 1;
            }
        }
    }
}

/*****************************************************************************************************/
/* This commented-out section describes our attempt to parallelize classifying samples to leaf nodes */
/*****************************************************************************************************/
// __global__ void cuda_leaf_assign(float* flat_data_table, float *tree,std::vector<int> *leaf_bins, int *leafAssignment)
// {
//  int nth_sample = blockDim.x * blockIdx.x + threadIdx.x;
//  int upper = pow(2, sizeof(tree)/sizeof(float)) - 1;
//  int lower = 0;
//  // perform binary search to classify sample
//  for(int nth_variable = 0; nth_variable < sizeof(tree)/sizeof(float); nth_variable ++)
//  {
//      if (nth_variable == sizeof(tree)/sizeof(float) - 1) // if we've reached the last decision node
//      {
//          if (flat_data_table[nth_sample*N_VARIABLES + nth_variable] <= tree[nth_variable])
//          {
//              leafAssignment[nth_sample] = lower;
//              leaf_bins[lower].push_back(nth_sample);
//          }
//          else
//          {
//              leafAssignment[nth_sample] = upper;
//              leaf_bins[upper].push_back(nth_sample);
//          }
//      }
//      if (flat_data_table[nth_sample*N_VARIABLES + nth_variable] <= tree[nth_variable])
//      {
//          upper = upper/2;
//      }
//      else
//      {
//          lower = upper/2;
//      }
//  }
// }

// run this on CPU. Initialize the array of predictions
void preprocessing(float * actual, float * predicted_array, CSVRow data_table[])
{
    float runningSum = 0;
    //  take the average of all elements in the output's row of data_table
    for (int i = 0; i < N_DATA; i++)
    {
            runningSum += data_table[i][N_VARIABLES];
    }
    float average = runningSum/N_DATA;

    // place the average into each spot in predicted_array
    for (int i = 0; i < N_DATA; i++)
    {
            memcpy(&predicted_array[i], &average, sizeof(float));
    }
}

static __inline__ uint64_t gettime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static uint64_t usec;

__attribute__ ((noinline))  void begin_roi() {
  usec=gettime();
}

__attribute__ ((noinline))  void end_roi()   {
  usec=(gettime()-usec);
  std::cout << "elapsed (sec): " << usec/1000000.0 << "\n";
}

// failed attempt to combine our smaller kernels to implement shared memory
__global__ void d_everything(int *d_leafBins, float *d_residual, float *d_leafValue, int *bins, float *d_predicted, int *d_leafAssignment, float lr, float *d_actual) {
    grid_group grid = this_grid();
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__  float s_residual[MAX_THREADS];
    __shared__ float s_actual[MAX_THREADS];
    __shared__ float s_predicted[MAX_THREADS];
    __shared__ float s_leafValue[MAX_THREADS];
    __shared__ int s_leafAssignment[MAX_THREADS];

    s_residual[threadIdx.x] = d_residual[i];
    s_actual[threadIdx.x] = d_actual[i];
    s_predicted[threadIdx.x] = d_predicted[i];
    s_leafAssignment[threadIdx.x] = d_leafAssignment[i];
    __syncthreads();
    grid.sync();
    for (int k = 0; k < ITERATIONS; k++) {
        // get residuals
        d_residual[threadIdx.x] = s_actual[threadIdx.x] - s_predicted[threadIdx.x];
        __syncthreads();
        grid.sync();
        // average bins
        if (i < 16) {
                int start;
            if (i == 0) {
                start = 0;
            } else {
                start = bins[i - 1];
            }
            int end = bins[i];

            for (int j = start; j < end; j++ ) {
                d_leafValue[i] += d_residual[d_leafBins[j]];
            }

            d_leafValue[i] /= end;
        }
        __syncthreads();
        grid.sync();

        while(lr*d_leafValue[s_leafAssignment[threadIdx.x]] < 1  || lr*d_leafValue[s_leafAssignment[threadIdx.x]] > -1)
        {
                // this loop is never taken, but get new predictions is still never updated...
        }

        // get new predictions
        s_predicted[threadIdx.x] += lr * d_leafValue[s_leafAssignment[threadIdx.x]];
        __syncthreads();
        grid.sync();
        // reset leaf vals
        s_leafValue[s_leafAssignment[threadIdx.x]] = 0;
        __syncthreads();
        grid.sync();
    }
    __syncthreads();
    grid.sync();
    memcpy(&d_predicted[i], &s_predicted[threadIdx.x], sizeof(float));
}

__global__ void d_averageBins(int *d_leafBins, float *d_residual, float *d_leafValue, int *bins) {
    int i = threadIdx.x;
    int start;

    d_leafValue[i] = 0.;
    if (i == 0) {
        start = 0;
    } else {
        start = bins[i - 1];
    }
    int end = bins[i];

    for (int j = start; j < end; j++ ) {
        d_leafValue[i] += d_residual[d_leafBins[j]];
    }

    d_leafValue[i] /= end;
}

__global__ void d_getNewPredictions(float *d_predicted, float *d_leafValue, int *d_leafAssignment, float lr) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d_predicted[i] += lr * d_leafValue[int(d_leafAssignment[i])];
}

__global__ void d_getNewResiduals(float *d_actual, float *d_predicted, float *d_residual) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d_residual[i] = d_actual[i] - d_predicted[i];
}

__global__ void d_resetLeafValues(float * d_leafValue) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    d_leafValue[i] = 0;
}

void averageBins(int*leafBins, float *residual, float *leafValue, int *bins) {
    int start, end;

    for (int i = 0; i < int(pow(2,N_VARIABLES)); i++) {

        leafValue[i] = 0.;
        if (i == 0) {
            start = 0;
        } else {
            start = bins[i - 1];
        }
        end = bins[i];

        for (int j = start; j < end; j++ ) {
            leafValue[i] += residual[leafBins[j]];
        }

        leafValue[i] /= end;
    }

}

void getNewPredictions(float *predicted, float *leafValue, int *leafAssignment, float lr) {
    for (int i = 0; i < N_DATA; i++) {
        predicted[i] += lr * leafValue[leafAssignment[i]];
    }
}

void getNewResiduals(float *actual, float *predicted, float *residual) {
    for (int i = 0; i < N_DATA; i++) {
        residual[i] = actual[i] - predicted[i];
    }
}

int main()
{
    std::ifstream file;
    file.open("/home/ericdang/code/3d.txt");
    CSVRow              variable;

    // data_table[i][j] corresponds to the ith data point and jth variable. If j = N_VARIABLES, j
    // is the output of the ith data point
    CSVRow              data_table[N_DATA];
    // float flat_data_table[102400*(N_VARIABLES+1)];
    int row = 0;
    while(file >> variable)
    {
        data_table[row] = variable;
        row++;
    }
    file.close();

    // initialize data
    std::vector<int> leafBins[int(pow(2, N_VARIABLES))] __attribute__((aligned(64)));
    int leafAssignment[N_DATA] __attribute__((aligned(64)));
    float tree[N_VARIABLES] __attribute__((aligned(64)));
    float residual[N_DATA] __attribute__((aligned(64)));
    float leafValue[int(pow(2, N_VARIABLES))] __attribute__((aligned(64)));
    float actual[N_DATA] __attribute__((aligned(64)));
    float predicted[N_DATA] __attribute__((aligned(64)));
    float resultGPU[N_DATA] __attribute__((aligned(64)));
    int flatLeafBins[N_DATA] __attribute__((aligned(64)));
    int bins[int(pow(2, N_VARIABLES))];

    // get actual values
    for (int i = 0; i < N_DATA; i++) {
        actual[i] = data_table[i][N_VARIABLES];
    }

    preprocessing(actual, predicted, data_table);
    initialize_tree(data_table, tree);
    std::fill_n(leafValue, int(pow(2, N_VARIABLES)), 0);
    std::fill_n(leafAssignment, N_DATA, 0);
    std::fill_n(residual, N_DATA, 0);
    leaf_assign(data_table, tree, leafBins, leafAssignment);

    // Allocate memory
    cudaError_t err = cudaSuccess;
    size_t size_output = N_DATA * sizeof(float);
    size_t size_bins = N_DATA * sizeof(int);
    size_t size_var = int(pow(2, N_VARIABLES)) * sizeof(float);

    // flatten leafbins and fill bins with stop values
    int progress = 0;
    for (int i = 0; i < int(pow(2, N_VARIABLES)); i++) {
        bins[i] = leafBins[i].size() + progress;
        std::cout << "bin value: " << bins[i] << "\n";

        for (int j = 0; j < leafBins[i].size(); j++) {
            flatLeafBins[progress + j] = leafBins[i][j];
        }
        progress += leafBins[i].size();
    }

    int *d_leafBins;
    int *d_leafAssignment;
    int *d_bins;
    // uncomment for GPU implementation of leaf_assign
    // float *d_data_table;
    // float *d_tree;
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

    // allocate d_bins memory
    err = cudaMalloc((void **)&d_bins, pow(2, N_VARIABLES) * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_bins (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // allocate d_leafBin memory
    err = cudaMalloc((void **)&d_leafBins, size_bins);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_leafBin (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // uncomment for GPU implementation of leaf_assign
    // // allocate d_data_table memory
    // err = cudaMalloc((void **)&d_data_table, size_input);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device vector d_data_table (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

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
    err = cudaMalloc((void **)&d_leafValue, size_var);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_leafValue (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // transfer memory associated with initalized vars from host to device
    err = cudaMemcpy(d_leafBins, flatLeafBins, size_bins, cudaMemcpyHostToDevice);
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

    err = cudaMemcpy(d_bins, bins, pow(2, N_VARIABLES) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // uncomment for GPU implementation of leaf_assign
    // err = cudaMemcpy(d_data_table, data_table, size_input, cudaMemcpyHostToDevice);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // uncomment for GPU implementation of leaf_assign
    // err = cudaMemcpy(d_tree, tree, size_output, cudaMemcpyHostToDevice);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

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

    // uncomment for GPU implementation of leaf_assign
    // err = cudaMemcpy(d_leafValue, leafValue, size_output, cudaMemcpyHostToDevice);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // compute on CPU
    std::cout << "Begin CPU calculations" << std::endl;
    begin_roi();
    for (int i = 0; i < ITERATIONS; i++) {
        getNewResiduals(actual, predicted, residual);
        averageBins(flatLeafBins, residual, leafValue, bins);
        getNewPredictions(predicted, leafValue, leafAssignment, LR);
        std::fill_n(leafValue, int(pow(2, N_VARIABLES)), 0);
    }
    end_roi();

    //compute numBlocks, numThreads
    int numBlocks = N_DATA / MAX_THREADS;
    int numThreads = pow(2, N_VARIABLES);


    //compute on GPU
    std::cout << "Begin GPU calculations" << std::endl;
    begin_roi();
    // uncomment for GPU implementation of leaf_assign
    // cuda_leaf_assign<<<numBlocks, MAX_THREADS>>>(d_data_table, d_tree, d_leafBins, d_leafAssignment);
    // cudaDeviceSynchronize();
    // d_everything<<<numBlocks, MAX_THREADS>>>(d_leafBins, d_residual, d_leafValue, d_bins, d_predicted, d_leafAssignment, LR, d_actual, int(pow(2, N_VARIABLES)));
    // cudaDeviceSynchronize();

    /*******************************************************************************************************
    /* Shared memory implementation commented out. Uses cooperative kernel launch to synchronize blocks
    /* within kernel call to exploit memory reuse across our smaller kernels
    // dim3 gridDim(numBlocks,1,1);
    // dim3 blockDim(MAX_THREADS,1,1);
    // void **arguments;
    // int arrayLen = 8;
    // arguments = (void**)malloc(arrayLen * sizeof(void*));
    
    // double* learningRate;
    // *learningRate = LR;
    // std::cout << *learningRate << std::endl;
    // arguments[0] = d_leafBins; 
    // arguments[1] = d_residual; 
    // arguments[2] = d_leafValue; 
    // arguments[3] = d_bins; 
    // arguments[4] = d_predicted;
    // arguments[5] = d_leafAssignment; 
    // arguments[6] = learningRate;
    // arguments[7] = d_actual;
    // cudaStream_t stream = 0;

    // size_t sharedMem = 49152;
    // std::cout<< "hi2" << std::endl;
    // cudaLaunchCooperativeKernel(
    //   (const void*)d_everything,
    //   gridDim,
    //   blockDim,
    //   arguments,
    //   sharedMem,
    //   stream
    // ) ;
    *******************************************************************************************************/
    
    for (int i = 0; i < ITERATIONS; i++) {
        d_getNewResiduals<<<numBlocks, MAX_THREADS>>>(d_actual, d_predicted, d_residual);
        cudaDeviceSynchronize();
        d_averageBins<<<1, numThreads>>>(d_leafBins, d_residual, d_leafValue, d_bins);
        cudaDeviceSynchronize();
        d_getNewPredictions<<<numBlocks, MAX_THREADS>>>(d_predicted, d_leafValue, d_leafAssignment, LR);
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

    compare(predicted, resultGPU, N_DATA);

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

    // uncomment for GPU implementation of leaf_assign
    // err = cudaFree(d_data_table);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to free device vector d_leafBins (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // uncomment for GPU implementation of leaf_assign
    // err = cudaFree(d_tree);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to free device vector d_leafBins (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

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