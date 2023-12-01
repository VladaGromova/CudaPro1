#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define BLOCK_SIZE 5
#define MATRIX_SIZE 10
#pragma hd_warning_disable

typedef struct {
  int width;
  int height;
  int stride;
  float* elements;
} Matrix;

__device__ float GetElement(const Matrix A, int row, int col) {
  return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) {
  A.elements[row * A.stride + col] = value;
}

float GetElementCPU(const Matrix A, int row, int col) {
  return A.elements[row * A.stride + col];
}

void SetElementCPU(Matrix A, int row, int col, float value) {
  A.elements[row * A.stride + col] = value;
}

 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C, unsigned long long* time) {
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    unsigned long long startTime = clock();
   for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);
    unsigned long long finishTime = clock();
    *time = (finishTime - startTime);
}

// Function to create and initialize a matrix
void InitializeMatrix(Matrix& mat, int width, int height) {
  mat.width = width;
  mat.height = height;
  mat.stride =
      width;  // Assuming a simple row-major layout where stride equals width
  mat.elements = new float[width * height];

  // Initializing matrix elements with some values (example: incrementing
  // values)
  float value = 1.0f;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      SetElementCPU(mat, i, j, value);
      value += 1.0f;
    }
  }
}

int main() {
  Matrix A, B, C;
  InitializeMatrix(A, MATRIX_SIZE,
                   MATRIX_SIZE);  // Example: creating a 3x3 matrix
  InitializeMatrix(B, MATRIX_SIZE, MATRIX_SIZE);
  InitializeMatrix(C, MATRIX_SIZE, MATRIX_SIZE);

  // Displaying matrix elements as an example
  std::cout << "Matrix A:" << std::endl;
  for (int i = 0; i < A.height; ++i) {
    for (int j = 0; j < A.width; ++j) {
      std::cout << GetElementCPU(A, i, j) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Matrix B:" << std::endl;
  for (int i = 0; i < B.height; ++i) {
    for (int j = 0; j < B.width; ++j) {
      std::cout << GetElementCPU(B, i, j) << " ";
    }
    std::cout << std::endl;
  }

  Matrix d_A, d_B, d_C;
  d_A.width = d_A.stride = A.width; 
  d_A.height = A.height;
  d_B.width = d_B.stride = B.width; 
  d_B.height = B.height;
  d_C.width = d_C.stride = C.width; 
  d_C.height = C.height;
  cudaMalloc(&d_A.elements, A.width * A.height * sizeof(float));
  cudaMemcpy(d_A.elements, A.elements, A.width * A.height * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&d_B.elements, B.width * B.height * sizeof(float));
  cudaMemcpy(d_B.elements, B.elements, B.width * B.height * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&d_C.elements, C.width * C.height * sizeof(float));

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 dimGrid((int) ceil((double)B.width / (double)dimBlock.x),
               (int) ceil((double)A.height /(double) dimBlock.y)); 

  unsigned long long time;
  unsigned long long* d_time;
  cudaMalloc(&d_time, sizeof(unsigned long long));

  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_time); 
  cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  std::cout<<"Time: "<<time<<'\n';
  cudaMemcpy(C.elements, d_C.elements, C.width * C.height * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "Matrix C:" << std::endl;
  for (int i = 0; i < C.height; ++i) {
    for (int j = 0; j < C.width; ++j) {
      std::cout << GetElementCPU(C, i, j) << " ";
    }
    std::cout << std::endl;
  }
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
  cudaFree(d_time);
  return 0;
}