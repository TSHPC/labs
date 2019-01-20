#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

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

  /// allocate memory in host RAM
  float *h_a = malloc(sizeof(float)*N*N);
  float *h_b = malloc(sizeof(float)*N*N);
  float *h_c = malloc(sizeof(float)*N*N);

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


  /// some events to count the execution time
  clock_t start = clock();
  cpu_matrix_mult(h_a, h_b, h_c, N);
  clock_t end = clock();
	
  double cpu_elapsed_time_ms = (end - start)/(double)CLOCKS_PER_SEC;

  printf("Time elapsed on CPU: %f ms.\n", cpu_elapsed_time_ms);

  /// free memory
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
