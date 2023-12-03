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
#include <limits>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#pragma hd_warning_disable
//#define FILENAME "data.txt"
//#define FILENAME "points_generated.txt"
#define FILENAME "myData.txt"
//#define FILENAME "cluster_data.txt"

#define BLOCK_SIZE 16
#define MAX_ITERATIONS 100
#define EPS 0.000001f
//#define MAX_THREADS_IN_BLOCK 512

#define MAX_THREADS_IN_BLOCK 16

struct EuclideanDistance {
    const float* special;
    int vectorSize;

    EuclideanDistance(const float* _special, int _vectorSize) : special(_special), vectorSize(_vectorSize) {}

    __host__ __device__
    float operator()(const thrust::tuple<const float*, const float*>& vec) const {
        float distance = 0.0f;
        for (int i = 0; i < vectorSize; ++i) {
            float diff = thrust::get<1>(vec)[i] - special[i];
            distance += diff * diff;
        }
        return sqrtf(distance);
    }
};

int main() {
  std::ifstream inputFile(FILENAME);
  std::string inputString;
  getline(inputFile, inputString);
  long N = atoi(inputString.c_str()); // real A height, real C height
  getline(inputFile, inputString);
  int n = atoi(inputString.c_str()); // real A width, real B height
  getline(inputFile, inputString);
  int k = atoi(inputString.c_str()); // real B width, real C width

  thrust::device_vector<float> collectionOfVectors((long)n * N); // Each vector represents a dimension
  thrust::device_vector<float> centroidsArray(n*k); // Each vector represents a dimension

  float value = 0.0f;
  thrust::device_vector<float> specialVector(n);
  int ind =0;
  while (getline(inputFile, inputString)) {
    std::istringstream iss(inputString);
    value = 0.0f;
    while (iss >> value) {
      collectionOfVectors.push_back(value);
      if(ind<n){
        specialVector.push_back(value);
      }
      ++ind;
    }
  }
  inputFile.close();

    std::cout << "Points Array:\n" << std::endl;
    for(int j = 0; j<N; ++j){
      for (int i = 0; i < n; ++i) {
        std::cout << collectionOfVectors[i] << " ";
      }
      std::cout << std::endl;
    }
    
    float* d_specialVector = thrust::raw_pointer_cast(specialVector.data());
    float* d_collectionOfVectors = thrust::raw_pointer_cast(collectionOfVectors.data());

    // Calculate distances in parallel using thrust::transform
    thrust::device_vector<float> distances(N);
     thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(specialVector.begin(), collectionOfVectors.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(specialVector.end(), collectionOfVectors.end())),
        distances.begin(),
        EuclideanDistance(d_specialVector, n)
    );

thrust::host_vector<float> distances_host = distances;

    // Print the distances
    std::cout << "Distances: ";
    for (int i = 0; i < N; ++i) {
        std::cout << distances_host[i] << " ";
    }
    std::cout << std::endl;
  std::cout<<"\nBye!\n";
  return 0;
}