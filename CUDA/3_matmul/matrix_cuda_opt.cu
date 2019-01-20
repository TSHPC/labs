#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

///
/// function name: gpu_matrix_mult
///
/// description: dot product of two matrix (not only square)
/// 
/// parameters: 
///            &a GPU device pointer to a m X n matrix (A)
///            &b GPU device pointer to a n X k matrix (B)
///            &c GPU device output purpose pointer to a m X k matrix (C) 
///            to store the result
///
/// Note:
///   grid and block should be configured as:
///   dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, 
///                (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
///   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
///
/// further speedup can be obtained by using shared memory to decrease 
/// global memory access times
/// return: none
///
__global__ void gpu_matrix_mult(float *d_a, float *d_b, 
                                float *d_result, int N) { 
  __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int idx;
  float tmp = 0.0;
  
  /// fill me in.
} 

/// 
/// function name: cpu_matrix_mult
/// 
/// description: dot product of two matrix (not only square) in CPU, 
///              for validating GPU results
/// 
/// parameters: 
///             &a CPU host pointer to a m X n matrix (A)
///             &b CPU host pointer to a n X k matrix (B)
///             &c CPU host output purpose pointer to a m X k matrix (C) 
///             to store the result
/// return: none
/// 
void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int N) {
  int i,j,k;
  #pragma omp parallel for schedule(dynamic,50) collapse(2) private(i,j,k) shared(h_a,h_b,h_result)
  for( j=0;j<N;j++)
    for(i=0;i<N;i++)
      for(k=0;k<N;k++)
        h_result[j*N+i] += h_a[j*N+k]*h_b[k*N+i];
}

///
/// function name: main
/// 
/// description: test and compare
/// 
/// parameters: 
///             none
/// 
/// return: none
///
int main(int argc, char const *argv[])
{
  int N=2048;
  /// Fixed seed for illustration.
  srand(3333);

  /// allocate memory in host RAM, h_cc is used to store CPU result
  float *h_a, *h_b, *h_c, *h_cc;
  cudaMallocHost((void **) &h_a, sizeof(float)*N*N);
  cudaMallocHost((void **) &h_b, sizeof(float)*N*N);
  cudaMallocHost((void **) &h_c, sizeof(float)*N*N);
  cudaMallocHost((void **) &h_cc, sizeof(float)*N*N);

  /// random initialize matrix A
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      h_a[j*N + i] = rand() % 1024;
    }
  }

  /// random initialize matrix B
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      h_b[j*N + i] = rand() % 1024;
    }
  }

  /// c = 0
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      h_c[j*N + i] = 0.0;
    }
  }

  float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

  /// some events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /// start to count execution time of GPU version
  cudaEventRecord(start, 0);
  
  /// Allocate memory space on the device 
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **) &d_a, sizeof(float)*N*N);
  cudaMalloc((void **) &d_b, sizeof(float)*N*N);
  cudaMalloc((void **) &d_c, sizeof(float)*N*N);

  /// copy matrix A and B from host to device memory
  cudaMemcpy(d_a, h_a, sizeof(int)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int)*N*N, cudaMemcpyHostToDevice);

  unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  /// Launch kernel 
  gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);    
  
  /// Transfer results from device to host 
  cudaMemcpy(h_c, d_c, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
    
  /// time counting terminate
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  /// compute time elapse on GPU computing
  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on GPU: %f ms.\n\n", gpu_elapsed_time_ms);

  /// start the CPU version
  cudaEventRecord(start, 0);

  cpu_matrix_mult(h_a, h_b, h_cc, N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on on CPU: %f ms.\n\n", cpu_elapsed_time_ms);

  /// validate results computed by GPU
  bool all_ok = true;
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if(fabs(h_c[j*N + i] - h_cc[j*N + i]) > 1.e-4) {
        all_ok = false;
      }
    }
  }

  /// roughly compute speedup
  if(all_ok) {
    printf("all results are correct!!!, speedup = %f\n", 
            cpu_elapsed_time_ms / gpu_elapsed_time_ms);
  } else {
    printf("incorrect results\n");
  }

  /// free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFreeHost(h_cc);
  return 0;
}
