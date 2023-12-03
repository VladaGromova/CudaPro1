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

struct SquaredDistanceFromConstant {
    float constantValue;
    SquaredDistanceFromConstant(float value) : constantValue(value) {}
    __host__ __device__
    float operator()(const float& x) const {
        float diff = x - constantValue;
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

thrust::device_vector<float> coords;


for (int j=0; j<k; ++j){
  std:: cout<< "\nFor "<< j<<" centroid: \n";
  for (int i = 0; i < n; ++i) {
    std::cout<<"coords??\n";
    for (size_t iii = 0; iii < pointsArray[i].size(); ++iii) {
            std::cout << pointsArray[i][iii] << " ";
        }
        std:: cout<<'\nConstnt '<<centroidsArray[i][j]<<'\n';
        float distance = thrust::transform_reduce(
            pointsArray[i].begin(),
            pointsArray[i].end(),
            SquaredDistanceFromConstant(centroidsArray[i][j]),
            0.0f,
            thrust::plus<float>()
        );
        distance = sqrt(distance);
        std:: cout<<distance<<' ';
    }
}

  std::cout<<"\nBye!\n";
  return 0;
}