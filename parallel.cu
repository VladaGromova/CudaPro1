#include <cfloat>
#include <climits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>

#pragma hd_warning_disable
//#define FILENAME "points_generated.txt"
#define FILENAME "data.txt"
#define BLOCK_SIZE 16
#define MAX_ITERATIONS 100
#define EPS 0.0001f
#define MAX_THREADS_IN_BLOCK 512

typedef struct {
  int width;
  int height;
  int realWidth;
  int realHeight;
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


void InitializeMatrices(Matrix& matA, int widthA, int heightA, int realWidthA, int realHeightA,
          Matrix& matB, int widthB, int heightB, int realWidthB, int realHeightB, std::istream& inputFile) {
  matA.width = widthA;
  matA.height = heightA;
  matA.realWidth = realWidthA;
  matA.realHeight = realHeightA;
  matA.stride = widthA;  // Assuming a simple row-major layout where stride equals width
  matA.elements = new float[widthA * heightA];

  matB.width = widthB;
  matB.height = heightB;
  matB.realWidth = realWidthB;
  matB.realHeight = realHeightB;
  matB.stride = widthB;  // Assuming a simple row-major layout where stride equals width
  matB.elements = new float[widthB * heightB];

  std::string inputString;
  std::string word;
  int i = 0;
  int j = 0;
  float value = 0.0f;
  while (getline(inputFile, inputString)) {
    std::istringstream iss(inputString);
    j = 0;
    while (iss >> value) {
      SetElementCPU(matA, i, j, value);
      if(i < realWidthB){
        SetElementCPU(matB, j, i, value);
      }
      ++j;
    }
    ++i;
  }
  value = 0.0f;
  for (i = 0; i < heightA; ++i) {
    for (j = realWidthA; j < widthA; ++j) {
      SetElementCPU(matA, i, j, value);
    }
  }
  for (i = realHeightA; i < heightA; ++i) {
    for (j = 0; j < widthA; ++j) {
      SetElementCPU(matA, i, j, value);
    }
  }
  for (i = 0; i < heightB; ++i) {
    for (j = realWidthB; j < widthB; ++j) {
      SetElementCPU(matB, i, j, value);
    }
  }
  for (i = realHeightB; i < heightB; ++i) {
    for (j = 0; j < widthB; ++j) {
      SetElementCPU(matB, i, j, value);
    }
  }
}

void InitializeMatrix(Matrix& mat, int width, int height, int realWidth, int realHeight) {
  mat.width = width;
  mat.height = height;
  mat.realWidth = realWidth;
  mat.realHeight = realHeight;
  mat.stride = width;  // Assuming a simple row-major layout where stride equals width
  mat.elements = new float[width * height];
  float value = FLT_MAX;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      SetElementCPU(mat, i, j, value);
    }
  }
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

__global__ void KmeansKernel(Matrix A, Matrix B, Matrix C, unsigned long long* time) {
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0.0;
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
        for (int e = 0; e < BLOCK_SIZE; ++e){
            Cvalue += pow(As[row][e] - Bs[e][col],2);
        }
        __syncthreads();
    }
    if(fabs(GetElement(Csub, row, col) - FLT_MAX) > EPS){
      SetElement(Csub, row, col, sqrt(Cvalue));
    }
    unsigned long long finishTime = clock();
    *time = (finishTime - startTime);
}

__global__ void MinInEachRow(Matrix C, int* result) {
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

__global__ void CompareArrays(const int* array1, const int* array2, int size, int* count) {
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

int main() {
  std::ifstream inputFile(FILENAME);
  std::string inputString;
  getline(inputFile, inputString);
  int N = atoi(inputString.c_str()); // real A height, real C height
  getline(inputFile, inputString);
  int n = atoi(inputString.c_str()); // real A width, real B height
  getline(inputFile, inputString);
  int k = atoi(inputString.c_str()); // real B width, real C width

    int A_width = n;
    int B_height = n;
    int A_height = N;
    int B_width = k; 
    if(n % BLOCK_SIZE != 0){
        A_width += (BLOCK_SIZE - (n % BLOCK_SIZE));
        B_height += (BLOCK_SIZE - (n % BLOCK_SIZE));
    }
    if(N % BLOCK_SIZE != 0){
        A_height += (BLOCK_SIZE - (N % BLOCK_SIZE));
    }
    if(k % BLOCK_SIZE != 0){
        B_width += (BLOCK_SIZE - (k % BLOCK_SIZE));
    }

  Matrix A, B, C;
  InitializeMatrices(A, A_width, A_height, n, N, B, B_width, B_height, k, n, inputFile);
  InitializeMatrix(C, B_width, A_height, k, N);
  inputFile.close();
  std::cout << "Matrix A:" << std::endl;
  for (int i = 0; i < A.realHeight; ++i) {
    for (int j = 0; j < A.realWidth; ++j) {
      std::cout << GetElementCPU(A, i, j) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Matrix B:" << std::endl;
  for (int i = 0; i < B.realHeight; ++i) {
    for (int j = 0; j < B.realWidth; ++j) {
      std::cout << GetElementCPU(B, i, j) << " ";
    }
    std::cout << std::endl;
  }
  Matrix d_A, d_B, d_C;
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

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 dimGrid((int) ceil((double)B.width / (double)dimBlock.x),
               (int) ceil((double)A.height /(double) dimBlock.y)); 

  unsigned long long time;
  unsigned long long* d_time;
  cudaMalloc(&d_time, sizeof(unsigned long long));

  int* assignments = new int[N];
  std::fill(assignments, assignments + N, 0);
  int* d_assignments;
  cudaMalloc(&d_assignments, N * sizeof(int));
  cudaMemcpy(d_assignments, assignments, N * sizeof(int), cudaMemcpyHostToDevice);

  int* newassignments = new int[N];
  std::fill(newassignments, newassignments + N, 0);
  int* d_newassignments;
  cudaMalloc(&d_newassignments, N * sizeof(int));
  cudaMemcpy(d_newassignments, newassignments, N * sizeof(int), cudaMemcpyHostToDevice);

int numIters = 0;
int changes = INT_MAX;
int* d_changes;
cudaMalloc(&d_changes, sizeof(int));
cudaMemset(d_changes, 0, sizeof(int));

int gridSize = C.realHeight/MAX_THREADS_IN_BLOCK + 1;
std::cout<<"gridSize: "<<gridSize<<'\n';

while(numIters < 1 && (float)changes/(float)N > EPS){
  KmeansKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_time); 
  MinInEachRow<<<gridSize, MAX_THREADS_IN_BLOCK>>>(d_C, d_newassignments);
  //CompareArrays<<<gridSize, MAX_THREADS_IN_BLOCK>>>(d_newassignments, d_assignments, N, d_changes);
  cudaMemcpy(newassignments, d_newassignments, N*sizeof(int), cudaMemcpyDeviceToHost); // optional
  cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
  ++numIters;
}
//std::cout << "\nNumber of different elements: " << changes << std::endl;
std::cout<< "Min in each row:\n";
for (int i=0; i<N; ++i) {
  std::cout<<newassignments[i]<<' ';
}
std:: cout<<'\n';

  cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  std::cout<<"Time: "<<time<<'\n';
  cudaMemcpy(C.elements, d_C.elements, C.width * C.height * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "Matrix C:" << std::endl;
  for (int i = 0; i < C.realHeight; ++i) {
    for (int j = 0; j < C.realWidth; ++j) {
      std::cout << GetElementCPU(C, i, j) << " ";
    }
    std::cout << std::endl;
  }
 

delete[] A.elements;
delete[] B.elements;
delete[] C.elements;
delete[] assignments;
delete[] newassignments;
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
  cudaFree(d_time);
  cudaFree(d_assignments);
  cudaFree(d_changes);
  std::cout<<"\nBye!\n";
  return 0;
}