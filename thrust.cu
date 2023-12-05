#include <cfloat>
#include <climits>
//#include <concurrencysal.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iterator>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <cstdio>
#include <math.h>

#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#pragma hd_warning_disable
#define FILENAME "data.txt"
//#define FILENAME "points_generated.txt"
//#define FILENAME "myData.txt"
//#define FILENAME "cluster_data.txt"

#define EPS 0.000001f

unsigned long long dtime_usec(unsigned long long prev){
#define USECPSEC 1000000ULL
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}

struct dkeygen : public thrust::unary_function<int, int>
{
  int dim;
  int numd;

  dkeygen(const int _dim, const int _numd) : dim(_dim), numd(_numd) {};

  __host__ __device__ int operator()(const int val) const {
    return (val/dim);
    }
};


struct clusterkeygen : public thrust::unary_function<int, int>
{
  int stride;

  clusterkeygen(const int _stride) : stride(_stride) {};

  __host__ __device__ int operator()(const int val) const {
    return (val % stride);
    }
};


typedef thrust::tuple<float, float> mytuple;
struct my_dist : public thrust::unary_function<mytuple, float>
{
  __host__ __device__ float operator()(const mytuple &my_tuple) const {
    float temp = thrust::get<0>(my_tuple) - thrust::get<1>(my_tuple);
    return temp*temp;
  }
};


struct MinWithIndex {
    __host__ __device__
    thrust::tuple<float, int> operator()(const thrust::tuple<float, int>& a, const thrust::tuple<float, int>& b) const {
        return (thrust::get<0>(a) < thrust::get<0>(b)) ? a : b;
    }
};


struct d_idx : public thrust::unary_function<int, int>
{
  int dim;
  int numd;

  d_idx(int _dim, int _numd) : dim(_dim), numd(_numd) {};

  __host__ __device__ int operator()(const int val) const {
    return (val % (dim*numd));
    }
};



struct c_idx : public thrust::unary_function<int, int>
{
  int dim;
  int numd;

  c_idx(int _dim, int _numd) : dim(_dim), numd(_numd) {};

  __host__ __device__ int operator()(const int val) const {
    return (val % dim) + (dim * (val/(dim*numd)));
    }
};

struct my_sqrt : public thrust::unary_function<float, float>
{
  __host__ __device__ float operator()(const float val) const {
    return sqrtf(val);
  }
};

template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
  T C; // number of columns
  
  __host__ __device__
  linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i)
  {
    return i / C;
  }
};

struct sum_functor {
    __host__ __device__
    float operator()(const float &a, const float &b) const {
        return a + b;
    }
};

struct div_functor : public thrust::unary_function<int,int>
{
    int m;
    div_functor(int _m) : m(_m) {};

    __host__ __device__
    int operator()(int x) const
    {
        return x / m;
    }
};

struct is_true
{
    __host__ __device__
    bool operator()(bool x) { return x; }
};

unsigned long long eucl_dist_thrust(thrust::host_vector<float> &cs, thrust::host_vector<float> &data, thrust::host_vector<float> &dist, int k, int n, int N, int print){

  thrust::device_vector<float> d_data = data;
  thrust::device_vector<float> d_centr = cs;
  thrust::device_vector<float> values_out(k*N);

  unsigned long long compute_time = dtime_usec(0);

thrust::reduce_by_key(
    // keys: 0...0 1...1 ... k*n*N
    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), dkeygen(n, N)), // begining of input key range
    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(n*N*k), dkeygen(n, N)), // end of input key range
    thrust::make_transform_iterator(thrust::make_zip_iterator( // begining of values range - tu chcemy miec odleglosci
      thrust::make_tuple(
        thrust::make_permutation_iterator(
          d_centr.begin(), 
          thrust::make_transform_iterator(
              thrust::make_counting_iterator<int>(0), d_idx(n, k)
          )
        ),
        thrust::make_permutation_iterator(
          d_data.begin(), 
          thrust::make_transform_iterator(
            thrust::make_counting_iterator<int>(0), c_idx(n, k)
          )
        )
      )
     ), my_dist()),
    thrust::make_discard_iterator(), // keys output (nie potrzebujemy tego)
    values_out.begin()    // values output - wynik
    );

  thrust::transform(values_out.begin(), values_out.end(), values_out.begin(), my_sqrt());
  cudaDeviceSynchronize();
 compute_time = dtime_usec(compute_time);
std:: cout<<"Distances :\n";
  if (print){
    thrust::copy(values_out.begin(), values_out.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;
    }
  thrust::copy(values_out.begin(), values_out.end(), dist.begin());

int numColumns = k; // Number of columns
    thrust::device_vector<float> mins(N);
    thrust::device_vector<int> d_clusters(N);

    // Perform reduction to find minimum value and its position for each row
    thrust::reduce_by_key(
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(numColumns)),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(k*N), linear_index_to_row_index<int>(numColumns)),
        thrust::make_zip_iterator(thrust::make_tuple(values_out.begin(), thrust::counting_iterator<int>(0))),
        thrust::make_discard_iterator(), // Discard keys output
        thrust::make_zip_iterator(thrust::make_tuple(mins.begin(), d_clusters.begin())),
        thrust::equal_to<int>(),
        MinWithIndex()
    );
    
thrust::device_vector<float> V2(N*k);
thrust::fill(V2.begin(), V2.end(), k);
thrust::transform(d_clusters.begin(), d_clusters.end(), V2.begin(), d_clusters.begin(), thrust::modulus<int>());

 std:: cout<<"\nClusters:\n";
   thrust::copy_n(d_clusters.begin(),d_clusters.end(),std::ostream_iterator<int>(std::cout, ", "));
   std::cout << std::endl;


 std:: cout<<"\nMins:\n";
   thrust::copy_n(mins.begin(),mins.end(),std::ostream_iterator<float>(std::cout, ", "));
   std::cout << std::endl;


  thrust::device_vector<int> indices(N);
    thrust::sequence(indices.begin(), indices.end());
    thrust::sort_by_key(d_clusters.begin(), d_clusters.end(), indices.begin());

 std:: cout<<"\nIndices:\n";
   thrust::copy_n(indices.begin(),indices.end(),std::ostream_iterator<float>(std::cout, ", "));
   std::cout << std::endl;

    // Oblicz liczbę wystąpień każdego klastra
    thrust::device_vector<int> clusterSizes(k);
    thrust::reduce_by_key(
        d_clusters.begin(), d_clusters.end(),
        thrust::make_constant_iterator(1),
        thrust::make_discard_iterator(),
        clusterSizes.begin(),
        thrust::equal_to<int>(),
        thrust::plus<int>()
    );
     std:: cout<<"\nCluster sizes:\n";
   thrust::copy_n(clusterSizes.begin(),clusterSizes.end(),std::ostream_iterator<float>(std::cout, ", "));
   std::cout << std::endl;


thrust::fill(d_centr.begin(), d_centr.end(), 0.0);
thrust::device_vector<float> vectorsInCluster(n);
thrust:: device_vector<float> actual_indices(1);
thrust::device_vector<int> data_starts(k);
thrust::device_vector<int> data_ends(k);
thrust::exclusive_scan(clusterSizes.begin(), clusterSizes.end(), data_starts.begin()); 
thrust::inclusive_scan(clusterSizes.begin(), clusterSizes.end(), data_ends.begin()); 
std:: cout<<"\n Data starts:\n";
thrust::copy_n(data_starts.begin(),data_starts.end(),std::ostream_iterator<int>(std::cout, ", "));
std::cout << std::endl;
std:: cout<<"\n Data ends:\n";
thrust::copy_n(data_ends.begin(),data_ends.end(),std::ostream_iterator<int>(std::cout, ", "));
std::cout << std::endl;
thrust::device_vector<bool> docopy(N*n);

int i=0;
//for(int i=0; i<k; ++i){
  vectorsInCluster.resize(clusterSizes[i] * n);
  actual_indices.resize(clusterSizes[i]);
  thrust::copy(indices.begin() + data_starts[i], indices.end() + data_ends[i], actual_indices.begin());

  typedef thrust::counting_iterator<int> counter;
    typedef thrust::transform_iterator<div_functor, counter> rowIterator;
    rowIterator rows_begin = thrust::make_transform_iterator(thrust::make_counting_iterator(0), div_functor(n));
    rowIterator rows_end = rows_begin + (N*n);
  thrust::binary_search(actual_indices.begin(), actual_indices.end(), rows_begin, rows_end, docopy.begin());
  thrust::copy_if(thrust::make_counting_iterator<int>(0), 
                  thrust::make_counting_iterator<int>(N*n),
                  docopy.begin(), 
                  vectorsInCluster.begin(), 
                  is_true()
  );
  std:: cout<<"\n Actrual vectors:\n";
thrust::copy_n(vectorsInCluster.begin(),vectorsInCluster.end(),std::ostream_iterator<float>(std::cout, ", "));
std::cout << std::endl;
//}


    // // Podziel sumę przez liczbę wystąpień, aby otrzymać centroidy
    // thrust::transform(
    //     centroids.begin(), centroids.end(),
    //     clusterSizes.begin(),
    //     centroids.begin(),
    //     thrust::divides<float>()
    // );


  return compute_time;
}

int main() {
  std::ifstream inputFile(FILENAME);
  std::string inputString;
  getline(inputFile, inputString);
  long N = atoi(inputString.c_str()); // real A height, real C height
  getline(inputFile, inputString);
  int n = atoi(inputString.c_str()); // real A width, real B height
  getline(inputFile, inputString);
  int k = atoi(inputString.c_str()); // real B width, real C width

  float* data = new float[N*n];
  float* centroids = new float[k*n];

  float value = 0.0f;
  int ind = 0;
  while (getline(inputFile, inputString)) {
    std::istringstream iss(inputString);
    value = 0.0f;
    while (iss >> value) {
      data[ind] = value;
      if(ind < k*n){
        centroids[ind]  = value;
      }
      ++ind;
    }
  }
  inputFile.close();
  std::cout<<"Data: \n";
  for (int i = 0; i<N; ++i) {
    for(int j=0; j< n; ++j){
      std::cout<< data[i*n + j]<< ' ';
    }
    std::cout<<'\n';
  }
    std::cout<<"\nCentroids: \n";
  for (int i = 0; i<k; ++i) {
    for(int j=0; j< n; ++j){
      std::cout<< data[i*n + j]<< ' ';
    }
    std::cout<<'\n';
  }
  
  thrust::host_vector<float> h_data(data, data + N*n);
  thrust::host_vector<float> h_centr(centroids, centroids + k*n);
  thrust::host_vector<float> h_dist(k*N);
  eucl_dist_thrust(h_centr, h_data, h_dist, k, n, N, 1);
    
  std::cout<<"\nBye!\n";
  return 0;
}