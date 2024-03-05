#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <sstream>
#include <stdlib.h>
#include <string>
//#include <sys/time.h>
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

#pragma hd_warning_disable

#define MAX_ITERATIONS 100
#define EPS 0.000001f


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
                        thrust::device_vector<float> &tmp_distances) {
  // we want to imitate this structure: (c - centroids, v - vetors)
  // c1 c2 ... ck | c1 c2 ... ck | ...
  // v1 v1 ... v1 | v2 v2 ... v2 | ...
  // => d(v1,c1) ... d(v1,cn) | ....
  // => min_dist_v1_cluster | min_dist_v2_cluster | ...
  thrust::reduce_by_key(
      // keys: 0...0 1...1 ... k*N
      thrust::make_transform_iterator(
          thrust::make_counting_iterator<int>(0),
          dkeygen(n, N)), // mod n (begining of input key range)
      thrust::make_transform_iterator(
          thrust::make_counting_iterator<int>(n * N * k),
          dkeygen(n, N)), // (end of input key range)
      thrust::make_transform_iterator(
          thrust::make_zip_iterator( // 
              thrust::make_tuple(
                  thrust::make_permutation_iterator(
                      d_centr.begin(),
                      thrust::make_transform_iterator(
                          thrust::make_counting_iterator<int>(0), d_idx(n, k))), // function to take coordinate of centroid
                  thrust::make_permutation_iterator(
                      d_data.begin(),
                      thrust::make_transform_iterator(
                          thrust::make_counting_iterator<int>(0), 
                          c_idx(n, k))))), // to take coordinate of vector 
          my_dist()),
      thrust::make_discard_iterator(), // keys output, we don't need it
      tmp_distances.begin()               // values output - result (distances)
  );

  thrust::transform(tmp_distances.begin(), tmp_distances.end(), tmp_distances.begin(),
                    my_sqrt());
}

void findNearestCentroid(int &k, int &N, thrust::device_vector<float> &d_centr,
                         thrust::device_vector<float> &tmp_distances,
                         thrust::device_vector<float> &mins,
                         thrust::device_vector<float> &vec_modulus_k,
                         thrust::device_vector<int> &d_clusters) {
  // if we interprate it lika a matrix the task will be just to find minimum for each row 
  thrust::reduce_by_key(
      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                      linear_index_to_row_index<int>(k)), // to get the number of element in row 
                                      //-> it gives us cluster number (but from the begining of array, not row)
      thrust::make_transform_iterator(thrust::counting_iterator<int>(k * N),
                                      linear_index_to_row_index<int>(k)),
      thrust::make_zip_iterator(thrust::make_tuple(
          tmp_distances.begin(), thrust::counting_iterator<int>(0))),
      thrust::make_discard_iterator(), // we don;'t need keys output
      thrust::make_zip_iterator(
          thrust::make_tuple(mins.begin(), d_clusters.begin())),
      thrust::equal_to<int>(), MinWithIndex());

  thrust::fill(vec_modulus_k.begin(), vec_modulus_k.end(), k);
  thrust::transform(d_clusters.begin(), d_clusters.end(), vec_modulus_k.begin(),
                    d_clusters.begin(), thrust::modulus<int>()); // to get real number of centroid 
}

void countClusterChanges(int &delta, thrust::device_vector<int> &old_d_clusters,
                         thrust::device_vector<int> &d_clusters) { // difference between previous assignment and actual
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
  // if in 1st cluster there are v0, v5, in 2nd - v1, ... then indices will be [0, 5, 1, ...]
  thrust::sort_by_key(d_clusters.begin(), d_clusters.end(), indices.begin());

  // num of vectors in each cluster
  thrust::reduce_by_key(d_clusters.begin(), d_clusters.end(),
                        thrust::make_constant_iterator(1), // we have to add 1 to sum if vector is in cluster
                        thrust::make_discard_iterator(), clusterSizes.begin(),
                        thrust::equal_to<int>(), thrust::plus<int>());
  
  thrust::fill(d_centr.begin(), d_centr.end(), 0.0);
  // for each cluster we want to find range of indices (vector numbers-ids) - actual_indices
  thrust::exclusive_scan(clusterSizes.begin(), clusterSizes.end(),
                         data_starts.begin()); 
  thrust::inclusive_scan(clusterSizes.begin(), clusterSizes.end(),
                         data_ends.begin());
  
  for (int i = 0; i < k; ++i) {
    vectorsInCluster.resize(clusterSizes[i] * n); // vectors in i-th cluster
    actual_indices.resize(clusterSizes[i]);
    thrust::copy(indices.begin() + data_starts[i], indices.end() + data_ends[i],
                 actual_indices.begin()); // cut actual segment
    
    thrust::binary_search(
        actual_indices.begin(), actual_indices.end(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                        div_functor(n)), // mod n
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                        div_functor(n)) +
            N * n,
        docopy.begin());
  
    thrust::copy_if(d_data.begin(), d_data.end(), docopy.begin(), // get actual vectors
                    vectorsInCluster.begin(), is_true());
    
    thrust::sequence(fcol_sums.begin(), fcol_sums.end());
    
    thrust::transform(
        fcol_sums.begin(), fcol_sums.end(), d_centr.begin() + i * n,
        centr_sum_functor(clusterSizes[i], n, // sum up elements from same columns
                          thrust::raw_pointer_cast(vectorsInCluster.data())));
    
    cudaDeviceSynchronize(); // reason: raw_pointer
    thrust::transform(d_centr.begin() + i * n, d_centr.begin() + (i + 1) * n,
                      thrust::make_constant_iterator(clusterSizes[i]),
                      d_centr.begin() + i * n, thrust::divides<float>());
    
  }
  
}


void KMeansClustering(float *&data, float *&cs, int *&clstrs, int k, int n,
                      int N, int print) {

  // additional data declaration
  thrust::device_vector<float> tmp_distances(k * N); // distances for k centroids and N vectors
  int delta = INT_MAX;
  int numIters = 0;
  thrust::device_vector<int> d_clusters(N); // vector for cluster assignments
  thrust::device_vector<int> old_d_clusters(N); // vector for previous cluster assignments
  thrust::fill(d_clusters.begin(), d_clusters.end(), 0);
  thrust::device_vector<float> mins(N); // minimum discance for each vector
  thrust::device_vector<float> vec_modulus_k(N * k); // vector filled with k to get the number of cluster from long array
  thrust::device_vector<int> indices(N); // id-s of vectors from sorted clusters
  thrust::device_vector<int> clusterSizes(k); // number of vectors in each cluster 
  thrust::device_vector<float> vectorsInCluster(n); // vectors in actual cluster (will be resized, but we always have >= 1)
  thrust::device_vector<float> actual_indices(1); // range from indices for actual cluster 
  thrust::device_vector<int> data_starts(k); // starts of segments from indices
  thrust::device_vector<int> data_ends(k); // ends of segments from indices
  thrust::device_vector<bool> docopy(N * n); // binary mask
  thrust::device_vector<float> fcol_sums(n);
  
  // CPU - GPU copying
  thrust::device_vector<float> d_data(data, data + n * N);
  thrust::device_vector<float> d_centr(cs, cs + n * k);

  while (numIters < MAX_ITERATIONS && (float)delta / (float)N > EPS) {
    delta = 0;
    
    // distance calculation
    calculateDistances(n, N, k, d_data, d_centr, tmp_distances);
   cudaDeviceSynchronize();
    // nearest centroid searching
    findNearestCentroid(k, N, d_centr, tmp_distances, mins, vec_modulus_k, d_clusters);
    
    // cluster changes counting
    countClusterChanges(delta, old_d_clusters, d_clusters);

    thrust::copy(d_clusters.begin(), d_clusters.end(),
                 old_d_clusters.begin()); // preprocessing

    
    // new centorids computation
    findNewCentroids(n, N, k, d_data, d_centr, indices, d_clusters,
                     clusterSizes, data_starts, data_ends, vectorsInCluster,
                     actual_indices, fcol_sums, docopy);
    ++numIters;
  }
  clstrs = new int[old_d_clusters.size()];
  thrust::copy(old_d_clusters.begin(), old_d_clusters.end(), clstrs);
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

void readFile(std::istream &inputFile, int &N, int &n, int &k, float *&data,
              float *&centroids) {
  std::string inputString;
  getline(inputFile, inputString);
  N = atoi(inputString.c_str()); 
  getline(inputFile, inputString);
  n = atoi(inputString.c_str()); 
  getline(inputFile, inputString);
  k = atoi(inputString.c_str()); 

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
  float *data; // vectors
  float *centroids; // new centroids
  int *clusters; // clusters[i] == old cluster number for i-th vector
  int N, n, k;
  readFile(inputFile, N, n, k, data, centroids);
  inputFile.close();

  // K-means clusterization
  KMeansClustering(data, centroids, clusters, k, n, N, 1);

  writeDataToFile(data, clusters, N, n);

  delete[] data;
  delete[] centroids;
  delete[] clusters;
  return 0;
}