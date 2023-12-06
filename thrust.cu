// #include <cfloat>
#include <climits>
// #include <concurrencysal.h>
// #include <cmath>
// #include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
// #include <iterator>
#include <limits.h>
// #include <math.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

// #include <time.h>

#pragma hd_warning_disable

#define MAX_ITERATIONS 100
#define EPS 0.000001f

unsigned long long dtime_usec(unsigned long long prev) {
#define USECPSEC 1000000ULL
  timeval tv1;
  gettimeofday(&tv1, 0);
  return ((tv1.tv_sec * USECPSEC) + tv1.tv_usec) - prev;
}

struct dkeygen : public thrust::unary_function<int, int> {
  int dim;
  int numd;

  dkeygen(const int _dim, const int _numd) : dim(_dim), numd(_numd){};

  __host__ __device__ int operator()(const int val) const {
    return (val / dim);
  }
};

struct clusterkeygen : public thrust::unary_function<int, int> {
  int stride;

  clusterkeygen(const int _stride) : stride(_stride){};

  __host__ __device__ int operator()(const int val) const {
    return (val % stride);
  }
};

typedef thrust::tuple<float, float> mytuple;
struct my_dist : public thrust::unary_function<mytuple, float> {
  __host__ __device__ float operator()(const mytuple &my_tuple) const {
    float temp = thrust::get<0>(my_tuple) - thrust::get<1>(my_tuple);
    return temp * temp;
  }
};

struct MinWithIndex {
  __host__ __device__ thrust::tuple<float, int>
  operator()(const thrust::tuple<float, int> &a,
             const thrust::tuple<float, int> &b) const {
    return (thrust::get<0>(a) < thrust::get<0>(b)) ? a : b;
  }
};

struct d_idx : public thrust::unary_function<int, int> {
  int dim;
  int numd;

  d_idx(int _dim, int _numd) : dim(_dim), numd(_numd){};

  __host__ __device__ int operator()(const int val) const {
    return (val % (dim * numd));
  }
};

struct c_idx : public thrust::unary_function<int, int> {
  int dim;
  int numd;

  c_idx(int _dim, int _numd) : dim(_dim), numd(_numd){};

  __host__ __device__ int operator()(const int val) const {
    return (val % dim) + (dim * (val / (dim * numd)));
  }
};

struct my_sqrt : public thrust::unary_function<float, float> {
  __host__ __device__ float operator()(const float val) const {
    return sqrtf(val);
  }
};

template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T, T> {
  T C; // number of columns

  __host__ __device__ linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__ T operator()(T i) { return i / C; }
};

struct sum_functor {
  __host__ __device__ float operator()(const float &a, const float &b) const {
    return a + b;
  }
};

struct div_functor : public thrust::unary_function<int, int> {
  int m;
  div_functor(int _m) : m(_m){};

  __host__ __device__ int operator()(int x) const { return x / m; }
};

struct is_true {
  __host__ __device__ bool operator()(bool x) { return x; }
};

struct centr_sum_functor {
  int R;
  int C;
  float *arr;

  centr_sum_functor(int _R, int _C, float *_arr) : R(_R), C(_C), arr(_arr){};

  __host__ __device__ float operator()(int myC) {
    float sum = 0.0;
    for (int i = 0; i < R; i++)
      sum += arr[i * C + myC];
    return sum;
  }
};

struct NotEqual {
  __host__ __device__ int operator()(thrust::tuple<int, int> t) const {
    return thrust::get<0>(t) != thrust::get<1>(t) ? 1 : 0;
  }
};

void calculateDistances(int &n, int &N, int &k,
                        thrust::device_vector<float> &d_data,
                        thrust::device_vector<float> &d_centr,
                        thrust::device_vector<float> &values_out) {
  thrust::reduce_by_key(
      // keys: 0...0 1...1 ... k*N
      thrust::make_transform_iterator(
          thrust::make_counting_iterator<int>(0),
          dkeygen(n, N)), // begining of input key range
      thrust::make_transform_iterator(
          thrust::make_counting_iterator<int>(n * N * k),
          dkeygen(n, N)), // end of input key range
      thrust::make_transform_iterator(
          thrust::make_zip_iterator( // begining of values range - tu
                                     // chcemy miec odleglosci
              thrust::make_tuple(
                  thrust::make_permutation_iterator(
                      d_centr.begin(),
                      thrust::make_transform_iterator(
                          thrust::make_counting_iterator<int>(0), d_idx(n, k))),
                  thrust::make_permutation_iterator(
                      d_data.begin(),
                      thrust::make_transform_iterator(
                          thrust::make_counting_iterator<int>(0),
                          c_idx(n, k))))),
          my_dist()),
      thrust::make_discard_iterator(), // keys output (nie potrzebujemy
                                       // tego)
      values_out.begin()               // values output - wynik
  );

  thrust::transform(values_out.begin(), values_out.end(), values_out.begin(),
                    my_sqrt());
}

void findNearestCentroid(int &k, int &N, thrust::device_vector<float> &d_centr,
                         thrust::device_vector<float> &values_out,
                         thrust::device_vector<float> &mins,
                         thrust::device_vector<float> &V2,
                         thrust::device_vector<int> &d_clusters) {
  thrust::reduce_by_key(
      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                      linear_index_to_row_index<int>(k)),
      thrust::make_transform_iterator(thrust::counting_iterator<int>(k * N),
                                      linear_index_to_row_index<int>(k)),
      thrust::make_zip_iterator(thrust::make_tuple(
          values_out.begin(), thrust::counting_iterator<int>(0))),
      thrust::make_discard_iterator(), // Discard keys output
      thrust::make_zip_iterator(
          thrust::make_tuple(mins.begin(), d_clusters.begin())),
      thrust::equal_to<int>(), MinWithIndex());

  thrust::fill(V2.begin(), V2.end(), k);
  thrust::transform(d_clusters.begin(), d_clusters.end(), V2.begin(),
                    d_clusters.begin(), thrust::modulus<int>());
}

void countClusterChanges(int &delta, thrust::device_vector<int> &old_d_clusters,
                         thrust::device_vector<int> &d_clusters) {
  delta = thrust::transform_reduce(
      thrust::make_zip_iterator(
          thrust::make_tuple(old_d_clusters.begin(), d_clusters.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(old_d_clusters.end(), d_clusters.end())),
      NotEqual(), 0, thrust::plus<int>());
}

void findNewCentroids(int &n, int &N, int &k,
                      thrust::device_vector<float> &d_data,
                      thrust::device_vector<float> &d_centr,
                      thrust::device_vector<int> &indices,
                      thrust::device_vector<int> &d_clusters,
                      thrust::device_vector<int> &clusterSizes,
                      thrust::device_vector<int> &data_starts,
                      thrust::device_vector<int> &data_ends,
                      thrust::device_vector<float> &vectorsInCluster,
                      thrust::device_vector<float> &actual_indices,
                      thrust::device_vector<float> &fcol_sums,
                      thrust::device_vector<bool> &docopy) {
  thrust::sequence(indices.begin(), indices.end());
  thrust::sort_by_key(d_clusters.begin(), d_clusters.end(), indices.begin());

  // num of vectors in each cluster
  thrust::reduce_by_key(d_clusters.begin(), d_clusters.end(),
                        thrust::make_constant_iterator(1),
                        thrust::make_discard_iterator(), clusterSizes.begin(),
                        thrust::equal_to<int>(), thrust::plus<int>());
  
  thrust::fill(d_centr.begin(), d_centr.end(), 0.0);
  thrust::exclusive_scan(clusterSizes.begin(), clusterSizes.end(),
                         data_starts.begin());
  thrust::inclusive_scan(clusterSizes.begin(), clusterSizes.end(),
                         data_ends.begin());
  
  for (int i = 0; i < k; ++i) {
    vectorsInCluster.resize(clusterSizes[i] * n);
    actual_indices.resize(clusterSizes[i]);
    thrust::copy(indices.begin() + data_starts[i], indices.end() + data_ends[i],
                 actual_indices.begin());
    
    thrust::binary_search(
        actual_indices.begin(), actual_indices.end(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                        div_functor(n)),
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                        div_functor(n)) +
            N * n,
        docopy.begin());
  
    thrust::copy_if(d_data.begin(), d_data.end(), docopy.begin(),
                    vectorsInCluster.begin(), is_true());
    
    thrust::sequence(fcol_sums.begin(), fcol_sums.end());
    
    thrust::transform(
        fcol_sums.begin(), fcol_sums.end(), d_centr.begin() + i * n,
        centr_sum_functor(clusterSizes[i], n,
                          thrust::raw_pointer_cast(vectorsInCluster.data())));
    
    //cudaDeviceSynchronize();
    thrust::transform(d_centr.begin() + i * n, d_centr.begin() + (i + 1) * n,
                      thrust::make_constant_iterator(clusterSizes[i]),
                      d_centr.begin() + i * n, thrust::divides<float>());
    
  }
  
}

void readFile(std::istream &inputFile, int &N, int &n, int &k, float *&data,
              float *&centroids) {
  std::string inputString;
  getline(inputFile, inputString);
  N = atoi(inputString.c_str()); // real A height, real C height
  getline(inputFile, inputString);
  n = atoi(inputString.c_str()); // real A width, real B height
  getline(inputFile, inputString);
  k = atoi(inputString.c_str()); // real B width, real C width

  data = new float[N * n];
  centroids = new float[k * n];
  float value = 0.0f;
  int ind = 0;
  while (getline(inputFile, inputString)) {
    std::istringstream iss(inputString);
    value = 0.0f;
    while (iss >> value) {
      data[ind] = value;
      if (ind < k * n) {
        centroids[ind] = value;
      }
      ++ind;
    }
  }
}

void eucl_dist_thrust(float *&data, float *&cs, int *&clstrs, int k, int n,
                      int N, int print) {

  // additional data declaration
  thrust::device_vector<float> values_out(k * N);
  int delta = INT_MAX;
  int numIters = 0;
  thrust::device_vector<int> d_clusters(N);
  thrust::device_vector<int> old_d_clusters(N);
  thrust::fill(d_clusters.begin(), d_clusters.end(), 0);
  thrust::device_vector<float> mins(N);
  thrust::device_vector<float> V2(N * k);
  thrust::device_vector<int> indices(N);
  thrust::device_vector<int> clusterSizes(k);
  thrust::device_vector<float> vectorsInCluster(n);
  thrust::device_vector<float> actual_indices(1);
  thrust::device_vector<int> data_starts(k);
  thrust::device_vector<int> data_ends(k);
  thrust::device_vector<bool> docopy(N * n);
  thrust::device_vector<float> fcol_sums(n);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  float tmpTime, elapsedTimeCalcDist = 0.0, elapsedTimeFindMin = 0.0,
                 elapsedTimeComapreArrays = 0.0,
                 elapsedTimeComputeAverage = 0.0;

  // CPU - GPU copying
  cudaEventRecord(start, 0);
  thrust::device_vector<float> d_data(data, data + n * N);
  thrust::device_vector<float> d_centr(cs, cs + n * k);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Elapsed Time [CPU - GPU copying] = " << elapsedTime
            << " milliseconds\n";

  while (numIters < MAX_ITERATIONS && (float)delta / (float)N > EPS) {
    delta = 0;
    
    // distance calculation
    cudaEventRecord(start, 0);
    calculateDistances(n, N, k, d_data, d_centr, values_out);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmpTime, start, stop);
    elapsedTimeCalcDist += tmpTime;

    
    // nearest centroid searching
    cudaEventRecord(start, 0);
    findNearestCentroid(k, N, d_centr, values_out, mins, V2, d_clusters);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmpTime, start, stop);
    elapsedTimeFindMin += tmpTime;

    
    // cluster changes counting
    cudaEventRecord(start, 0);
    countClusterChanges(delta, old_d_clusters, d_clusters);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmpTime, start, stop);
    elapsedTimeComapreArrays += tmpTime;

    thrust::copy(d_clusters.begin(), d_clusters.end(),
                 old_d_clusters.begin()); // preprocessing

    
    // new centorids computation
    cudaEventRecord(start, 0);
    findNewCentroids(n, N, k, d_data, d_centr, indices, d_clusters,
                     clusterSizes, data_starts, data_ends, vectorsInCluster,
                     actual_indices, fcol_sums, docopy);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmpTime, start, stop);
    elapsedTimeComputeAverage += tmpTime;

 
    ++numIters;
  }
  clstrs = new int[old_d_clusters.size()];
  thrust::copy(old_d_clusters.begin(), old_d_clusters.end(), clstrs);
  std::cout << "Iterations:" << numIters << '\n';
  std::cout << "Elapsed Time [Distance calculation stage] = "
            << elapsedTimeCalcDist << " milliseconds\n";
  std::cout << "Elapsed Time [Finding minimum stage] = " << elapsedTimeFindMin
            << " milliseconds\n";
  std::cout << "Elapsed Time [Array comparing stage] = "
            << elapsedTimeComapreArrays << " milliseconds\n";
  std::cout << "Elapsed Time [Computing average stage] = "
            << elapsedTimeComputeAverage << " milliseconds\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void writeDataToFile(float* data, const int* clusters, int N, int n) {
    std::ofstream outputFile;
    outputFile.open("out_thrust.txt");
    if (outputFile.is_open()) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < n; ++j) {
                outputFile << data[i*n + j] << ' ';
            }
            outputFile << clusters[i] << '\n';
        }
        outputFile.close();
        std::cout << "Data written successfully \n" << std::endl;
    } else {
        std::cout << "Unable to open the file \n"<< std::endl;
    }
}

int main(int argc, char **argv) {
  // file validation
  std::string inFile = "";
  if (argc == 2) {
    inFile = argv[1];
  } else {
    std::cout << "Usage: ./cufile InputFile \n";
    return 1;
  }
  std::ifstream inputFile;
  inputFile.open(inFile.c_str(), std::ios::in);
  if (!inputFile.is_open()) {
    std::cout << "Error opening file: " << inFile << std::endl;
    return 1;
  }

  // data declaration
  float *data;
  float *centroids;
  int *clusters;
  int N, n, k;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // data initialization from file
  cudaEventRecord(start, 0);
  readFile(inputFile, N, n, k, data, centroids);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "\nElapsed Time [Data reading] = " << elapsedTime
            << " milliseconds\n";
  inputFile.close();

  // K-means clusterization
  cudaEventRecord(start, 0);
  eucl_dist_thrust(data, centroids, clusters, k, n, N, 1);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Elapsed Time [Full algorithm + time measurement] = "
            << elapsedTime << " milliseconds\n";

  // std::cout << "points and clusters:\n";
  // for (int i = 0; i < N; ++i) {
  //   for (int j = 0; j < n; ++j) {
  //     std::cout << data[i * n + j] << ' ';
  //   }
  //   std::cout << clusters[i] << '\n';
  // }
  writeDataToFile(data, clusters, N, n);

  delete[] data;
  delete[] centroids;
  delete[] clusters;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}