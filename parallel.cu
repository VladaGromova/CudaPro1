#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>


#pragma hd_warning_disable

#define BLOCK_SIZE 16
#define MAX_ITERATIONS 100
#define EPS 0.000001f
#define MAX_THREADS_IN_BLOCK 512
//#define MAX_THREADS_IN_BLOCK 16

typedef struct {
  int width;
  int height;
  int realWidth;
  int realHeight;
  int stride;
  float *elements;
} Matrix;

__host__ __device__ float GetElement(const Matrix A, int row, int col) {
  return A.elements[row * A.stride + col];
}

__host__ __device__ void SetElement(Matrix A, int row, int col, float value) {
  A.elements[row * A.stride + col] = value;
}

void FillMatrices(Matrix &matA, int widthA, int heightA, int realWidthA,
                        int realHeightA, Matrix &matB, int widthB, int heightB,
                        int realWidthB, int realHeightB,
                        std::istream &inputFile) {
  matA.width = widthA;
  matA.height = heightA;
  matA.realWidth = realWidthA;
  matA.realHeight = realHeightA;
  matA.stride = widthA; // Assuming a row-major layout, stride == width
  matA.elements = new float[widthA * heightA];

  matB.width = widthB;
  matB.height = heightB;
  matB.realWidth = realWidthB;
  matB.realHeight = realHeightB;
  matB.stride = widthB; // Assuming a row-major layout, stride == width
  matB.elements = new float[widthB * heightB];

  std::string inputString;
  int i = 0;
  int j = 0;
  float value = 0.0f;
  while (getline(inputFile, inputString)) {
    std::istringstream iss(inputString);
    j = 0;
    while (iss >> value) {
      SetElement(matA, i, j, value);
      if (i < realWidthB) {
        SetElement(matB, j, i, value);
      }
      ++j;
    }
    ++i;
  }
  value = 0.0f;
  for (i = 0; i < heightA; ++i) {
    for (j = realWidthA; j < widthA; ++j) {
      SetElement(matA, i, j, value);
    }
  }
  for (i = realHeightA; i < heightA; ++i) {
    for (j = 0; j < widthA; ++j) {
      SetElement(matA, i, j, value);
    }
  }
  for (i = 0; i < heightB; ++i) {
    for (j = realWidthB; j < widthB; ++j) {
      SetElement(matB, i, j, value);
    }
  }
  for (i = realHeightB; i < heightB; ++i) {
    for (j = 0; j < widthB; ++j) {
      SetElement(matB, i, j, value);
    }
  }
}

void InitializeMatrix(Matrix &mat, int width, int height, int realWidth,
                      int realHeight) {
  mat.width = width;
  mat.height = height;
  mat.realWidth = realWidth;
  mat.realHeight = realHeight;
  mat.stride =
      width; // Assuming a row-major layout, stride == width
  mat.elements = new float[width * height];
  float value = FLT_MAX;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      SetElement(mat, i, j, value);
    }
  }
}

void InitializeDeviceMatrices(Matrix &A, Matrix &B, Matrix &C, Matrix &d_A, Matrix &d_B, Matrix &d_C){
  d_A.width = d_A.stride = A.width;
  d_A.height = A.height;
  d_A.realWidth = A.realWidth;
  d_A.realHeight = d_A.realHeight;
  d_B.width = d_B.stride = B.width;
  d_B.height = B.height;
  d_B.realWidth = B.realWidth;
  d_B.realHeight = B.realHeight;
  d_C.width = d_C.stride = C.width;
  d_C.height = C.height;
  d_C.realWidth = C.realWidth;
  d_C.realHeight = C.realHeight;
  cudaMalloc(&d_A.elements, A.width * A.height * sizeof(float));
  cudaMemcpy(d_A.elements, A.elements, A.width * A.height * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&d_B.elements, B.width * B.height * sizeof(float));
  cudaMemcpy(d_B.elements, B.elements, B.width * B.height * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&d_C.elements, C.width * C.height * sizeof(float));
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
  Matrix Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}

__global__ void CalculateDistances(Matrix A, Matrix B, Matrix C) {

  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
  float Cvalue = 0.0;
  int row = threadIdx.y;
  int col = threadIdx.x;

  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    Matrix Asub = GetSubMatrix(A, blockRow, m);
    Matrix Bsub = GetSubMatrix(B, m, blockCol);
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);
    __syncthreads();
    for (int e = 0; e < BLOCK_SIZE; ++e) {
      Cvalue += pow(As[row][e] - Bs[e][col], 2);
    }
    __syncthreads();
  }
  if (fabs(GetElement(Csub, row, col) - FLT_MAX) > EPS) {
    SetElement(Csub, row, col, sqrt(Cvalue));
  }
}

__global__ void MinInEachRow(Matrix C, int *result) {
  int rows = C.realHeight;
  int tid = threadIdx.x + blockIdx.x * blockDim.x; // nr wiersza
  float minValue;
  int minIndex;
  if (tid < rows) {
    minValue = GetElement(C, tid, 0);
    minIndex = 0;
    for (int j = 0; j < C.realWidth; ++j) {
      if (GetElement(C, tid, j) < minValue) {
        minValue = GetElement(C, tid, j);
        minIndex = j;
      }
    }
    result[tid] = minIndex;
  }
}

__global__ void CompareArrays(const int *array1, const int *array2, int size,
                              int *count) {
  __shared__ int localCounts[MAX_THREADS_IN_BLOCK];

  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int gridSize = blockDim.x * gridDim.x;

  localCounts[tid] = 0;
  for (int i = idx; i < size; i += gridSize) {
    if (array1[i] != array2[i]) {
      localCounts[tid]++;
    }
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      localCounts[tid] += localCounts[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(count, localCounts[0]);
  }
}

__global__ void ComputeAverage(Matrix B, const int *numOfVectors, int k,
                               int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < k) {
    for (int i = 0; i < n; ++i) {
      SetElement(B, i, tid,
                 (float)GetElement(B, i, tid) / (float)numOfVectors[tid]);
    }
  }
}

__global__ void ComputeSum(Matrix matA, const int *groups, Matrix matB, int N,
                           int k, int n, int *numOfVectors) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    int groupId = groups[tid];
    atomicAdd(&numOfVectors[groupId], 1);
    __syncthreads();
    for (int i = 0; i < n; ++i) {
      atomicAdd(&matB.elements[i * matB.stride + groupId],
                GetElement(matA, tid, i));
    }
  }
  __syncthreads();
}

void readFile(std::istream &inputFile, int& N, int& n, int& k, Matrix& A, Matrix& B, Matrix& C){
  std::string inputString;
  getline(inputFile, inputString);
  N = atoi(inputString.c_str()); // real A height, real C height
  getline(inputFile, inputString);
  n = atoi(inputString.c_str()); // real A width, real B height
  getline(inputFile, inputString);
  k = atoi(inputString.c_str()); // real B width, real C width

  // A is N*n, but I want be able to split A into full blocks, so I want the height and the width be divisible by BLOCK_SIZE
  // Same for B (n*k) and C (N*k)
  int A_width = n;
  int B_height = n;
  int A_height = N;
  int B_width = k;
  if (n % BLOCK_SIZE != 0) {
    A_width += (BLOCK_SIZE - (n % BLOCK_SIZE));
    B_height += (BLOCK_SIZE - (n % BLOCK_SIZE));
  }
  if (N % BLOCK_SIZE != 0) {
    A_height += (BLOCK_SIZE - (N % BLOCK_SIZE));
  }
  if (k % BLOCK_SIZE != 0) {
    B_width += (BLOCK_SIZE - (k % BLOCK_SIZE));
  }


  // Read data into matrices
  FillMatrices(A, A_width, A_height, n, N, B, B_width, B_height, k, n,
                     inputFile);
  // Matrix A contains dataset: one row - one vektor
  // Matrix B contains k centroids (first k vectors from dataset): one column - one vector 
  InitializeMatrix(C, B_width, A_height, k, N); // C will contain distances
}

void defineArray(int& N, int& k, int*& assignments, int*& d_assignments, int*& numOfVectorsInClusters, int*& d_numOfVectorsInClusters){ 
  assignments = new int[N];
  std::fill(assignments, assignments + N, 0);
  cudaMalloc(&d_assignments, N * sizeof(int));
  cudaMemcpy(d_assignments, assignments, N * sizeof(int),
             cudaMemcpyHostToDevice);

  newassignments = new int[N];
  std::fill(newassignments, newassignments + N, 0);
  cudaMalloc(&d_newassignments, N * sizeof(int));
  cudaMemcpy(d_newassignments, newassignments, N * sizeof(int),
             cudaMemcpyHostToDevice);

  numOfVectorsInClusters = new int[k];
  std::fill(numOfVectorsInClusters, numOfVectorsInClusters + k, 0);
  cudaMalloc(&d_numOfVectorsInClusters, k * sizeof(int));
  cudaMemset(d_numOfVectorsInClusters, 0, k * sizeof(int));
}

int main(int argc, char** argv) {
  std::string inFile = "";
    if( argc == 2 ) {
      inFile = argv[1];
    }
    else {
      std::cout << "Usage: ./cufile InputFile \n";
      return 1;
    }
  std::ifstream inputFile;
  inputFile.open(inFile.c_str(), std::ios::in);
  if (!inputFile.is_open()) {
        std::cout << "Error opening file: " << inFile << std::endl;
        return 1;
    }

  Matrix A, B, C; 
  int N; 
  int n;
  int k;

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  readFile(inputFile, N, n, k, A, B, C);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  std::cout<<"\n[Data reading] Elapsed Time = "<<elapsedTime<<" milliseconds\n";
  inputFile.close();

  Matrix d_A, d_B, d_C;
  cudaEventRecord(start,0);
  InitializeDeviceMatrices(A, B, C, d_A, d_B, d_C);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  std::cout<<"[CPU - GPU copying] Elapsed Time = "<<elapsedTime<<" milliseconds\n";


  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((int)ceil((double)B.width / (double)dimBlock.x),
               (int)ceil((double)A.height / (double)dimBlock.y));

  
  // int *assignments = new int[N];
  // std::fill(assignments, assignments + N, 0);
  // int *d_assignments;
  // cudaMalloc(&d_assignments, N * sizeof(int));
  // cudaMemcpy(d_assignments, assignments, N * sizeof(int),
  //            cudaMemcpyHostToDevice);
  int *assignments, *d_assignments, *newassignments, *d_newassignments,  *numOfVectorsInClusters, *d_numOfVectorsInClusters;
  defineArray(N, k, assignments, d_assignments, newassignments, d_newassignments, numOfVectorsInClusters, d_numOfVectorsInClusters);

  // int *newassignments = new int[N];
  // std::fill(newassignments, newassignments + N, 0);
  // int *d_newassignments;
  // cudaMalloc(&d_newassignments, N * sizeof(int));
  // cudaMemcpy(d_newassignments, newassignments, N * sizeof(int),
  //            cudaMemcpyHostToDevice);

  // int *numOfVectorsInClusters = new int[k];
  // std::fill(numOfVectorsInClusters, numOfVectorsInClusters + k, 0);
  // int *d_numOfVectorsInClusters;
  // cudaMalloc(&d_numOfVectorsInClusters, k * sizeof(int));
  // cudaMemset(d_numOfVectorsInClusters, 0, k * sizeof(int));

  int numIters = 0;
  int changes = INT_MAX;
  int *d_changes;
  cudaMalloc(&d_changes, sizeof(int));
  cudaMemset(d_changes, 0, sizeof(int));

  int gridSize = C.realHeight / MAX_THREADS_IN_BLOCK + 1;

  while (numIters < MAX_ITERATIONS && (float)changes / (float)N > EPS) {
    CalculateDistances<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaMemset(d_B.elements, 0.0, d_B.height * d_B.width * sizeof(float));
    cudaMemset(d_numOfVectorsInClusters, 0, k * sizeof(int));
    cudaMemset(d_changes, 0, sizeof(int));
    MinInEachRow<<<gridSize, MAX_THREADS_IN_BLOCK>>>(d_C, d_newassignments);
    CompareArrays<<<gridSize, MAX_THREADS_IN_BLOCK>>>(
        d_newassignments, d_assignments, N, d_changes);
    ComputeSum<<<gridSize, MAX_THREADS_IN_BLOCK>>>(
        d_A, d_newassignments, d_B, N, k, n, d_numOfVectorsInClusters);
    ComputeAverage<<<gridSize, MAX_THREADS_IN_BLOCK>>>(
        d_B, d_numOfVectorsInClusters, k, n);
    cudaMemcpy(d_assignments, d_newassignments, N * sizeof(int),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
    ++numIters;

    // optional
    cudaMemcpy(C.elements, d_C.elements, C.width * C.height * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(B.elements, d_B.elements, B.width * B.height * sizeof(float),
               cudaMemcpyDeviceToHost);
 
  }

  cudaMemcpy(B.elements, d_B.elements, B.width * B.height * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout<<"Iterations:"<< numIters<<'\n';
  std::cout << "Centroids:" << std::endl;
  for (int i = 0; i < B.realHeight; ++i) {
    for (int j = 0; j < B.realWidth; ++j) {
      std::cout << GetElement(B, i, j) << " ";
    }
    std::cout << std::endl;
  }
  
  delete[] A.elements;
  delete[] B.elements;
  delete[] C.elements;
  delete[] assignments;
  delete[] newassignments;
  delete[] numOfVectorsInClusters;
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
  cudaFree(d_assignments);
  cudaFree(d_newassignments);
  cudaFree(d_numOfVectorsInClusters);
  cudaFree(d_changes);
  std::cout << "Bye!\n";
  return 0;
}