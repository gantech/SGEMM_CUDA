#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * A stand-alone script to invoke & benchmark standard cuBLAS SGEMM performance
 */

int main(int argc, char *argv[]) {
  int m = 2;
  int k = 3;
  int n = 4;
  int print = 1;
  cudaError_t cudaStat;  // cudaMalloc status
  cublasStatus_t stat;   // cuBLAS functions status
  cublasHandle_t handle; // cuBLAS context

  int i, j;

  double *a, *b, *c;

  // malloc for a,b,c...
  a = (double *)malloc(m * k * sizeof(double));
  b = (double *)malloc(k * n * sizeof(double));
  c = (double *)malloc(m * n * sizeof(double));

  int ind = 11;
  for (j = 0; j < m * k; j++) {
    a[j] = (double)ind++;
  }

  ind = 11;
  for (j = 0; j < k * n; j++) {
    b[j] = (double)ind++;
  }

  ind = 11;
  for (j = 0; j < m * n; j++) {
    c[j] = (double)ind++;
  }

  // DEVICE
  double *d_a, *d_b, *d_c;

  // cudaMalloc for d_a, d_b, d_c...
  cudaMalloc((void **)&d_a, m * k * sizeof(double));
  cudaMalloc((void **)&d_b, k * n * sizeof(double));
  cudaMalloc((void **)&d_c, m * n * sizeof(double));

  stat = cublasCreate(&handle); // initialize CUBLAS context

  cudaMemcpy(d_a, a, m * k * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, k * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, m * n * sizeof(double), cudaMemcpyHostToDevice);

  double alpha = 1.0f;
  double beta = 0.5f;

  if (print == 1) {
    printf("alpha = %4.0f, beta = %4.0f\n", alpha, beta);
    printf("A = (mxk: %d x %d)\n", m, k);
    for (i = 0; i < m; i++) {
      for (j = 0; j < k; j++) {
        printf("%4.1f ", a[i * m + j]);
      }
      printf("\n");
    }
    printf("B = (kxn: %d x %d)\n", k, n);
    for (i = 0; i < k; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", b[i * n + j]);
      }
      printf("\n");
    }
    printf("C = (mxn: %d x %d)\n", m, n);
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n,
                     d_a, k, &beta, d_c, n);

  cudaMemcpy(c, d_c, m * n * sizeof(double), cudaMemcpyDeviceToHost);

  if (print == 1) {
    printf("\nC after SGEMM = \n");
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle); // destroy CUBLAS context
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
