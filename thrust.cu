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
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

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

struct SquaredDistance {
    int n; // Number of dimensions

    SquaredDistance(int _n) : n(_n) {}

    __host__ __device__
    float operator()(const float& x, const float& y) const {
        float diff = x - y;
        return diff * diff;
    }
};

int main() {
  std::ifstream inputFile(FILENAME);
  std::string inputString;
  getline(inputFile, inputString);
  int N = atoi(inputString.c_str()); // real A height, real C height
  getline(inputFile, inputString);
  int n = atoi(inputString.c_str()); // real A width, real B height
  getline(inputFile, inputString);
  int k = atoi(inputString.c_str()); // real B width, real C width

  std::vector<thrust::device_vector<float> > pointsArray(n); // Each vector represents a dimension
  std::vector<thrust::device_vector<float> > centroidsArray(n); // Each vector represents a dimension

  float value = 0.0f;
  int index = 0;
  int indexOfVector = 0;
  while (getline(inputFile, inputString)) {
    std::istringstream iss(inputString);
    value = 0.0f;
    index = 0;
    while (iss >> value) {
      pointsArray[index].push_back(value);
      if(indexOfVector < k){
        centroidsArray[index].push_back(value);
      }
      ++index;
    }
    ++indexOfVector;
  }

  inputFile.close();
    std::cout << "Points Array:" << std::endl;
    for (int i = 0; i < n; ++i) {
      std::cout<<"x_"<<i<<": ";
        for (size_t j = 0; j < pointsArray[i].size(); ++j) {
            std::cout << pointsArray[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nCentroids Array:" << std::endl;
    for (int i = 0; i < n; ++i) {
      std::cout<<"x_"<<i<<": ";
        for (size_t j = 0; j < centroidsArray[i].size(); ++j) {
            std::cout << centroidsArray[i][j] << " ";
        }
        std::cout << std::endl;
    }

   std::vector<thrust::device_vector<float> > distToCentroids(k); // Each vector represents a dimension

   for (int i = 0; i < n; ++i) {
        float distance = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(pointsArray[i].begin(), centroidsArray[i].begin())),
            thrust::make_zip_iterator(thrust::make_tuple(pointsArray[i].end(), centroidsArray[i].end())),
            SquaredDistance(n),
            0.0f,
            thrust::plus<float>()
        );
        distance = sqrt(distance);
        std:: cout<<distance<<' ';
        // Use the calculated distance as needed
        // For example, store or perform further operations with distances
    }

  std::cout<<"\nBye!\n";
  return 0;
}