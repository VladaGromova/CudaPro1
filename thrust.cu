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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/copy.h>
#include <math.h>

#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#pragma hd_warning_disable
//#define FILENAME "data.txt"
//#define FILENAME "points_generated.txt"
#define FILENAME "myData.txt"
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

typedef thrust::tuple<float, float> mytuple;
struct my_dist : public thrust::unary_function<mytuple, float>
{
  __host__ __device__ float operator()(const mytuple &my_tuple) const {
    float temp = thrust::get<0>(my_tuple) - thrust::get<1>(my_tuple);
    return temp*temp;
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

unsigned long long eucl_dist_thrust(thrust::host_vector<float> &centroids, thrust::host_vector<float> &data, thrust::host_vector<float> &dist, int k, int n, int N, int print){

  thrust::device_vector<float> d_data = data;
  thrust::device_vector<float> d_centr = centroids;
  thrust::device_vector<float> values_out(k*N);

  unsigned long long compute_time = dtime_usec(0);
auto result_tuple = thrust::make_tuple(
        thrust::make_permutation_iterator(
            d_centr.begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator<int>(0), c_idx(n, N)
            )
        ),
        thrust::make_permutation_iterator(
            d_data.begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator<int>(0), d_idx(n, N)
            )
        )
    );

    // Wypisywanie wynikowej tupli
    for (size_t i = 0; i < d_data.size(); ++i) {
        std::cout << "(" << thrust::get<0>(result_tuple)[i] << ", " << thrust::get<1>(result_tuple)[i] << ") ";
    }
    

  thrust::reduce_by_key(
    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), dkeygen(n, N)), // begining of input key range
    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(n*N*k), dkeygen(n, N)), // end of input key range
    thrust::make_transform_iterator(thrust::make_zip_iterator(
      thrust::make_tuple(
        thrust::make_permutation_iterator(
          d_centr.begin(), 
          thrust::make_transform_iterator(
              thrust::make_counting_iterator<int>(0), c_idx(n, N)
          )
        ),
        thrust::make_permutation_iterator(
          d_data.begin(), 
          thrust::make_transform_iterator(
            thrust::make_counting_iterator<int>(0), d_idx(n, N)
          )
        )
      )
     ), my_dist()),
    thrust::make_discard_iterator(), // keys output (nie potrzebujemy tego)
    values_out.begin()    // values outpu - wynik
    );
      cudaDeviceSynchronize();
    std:: cout<<"Values (1)\n";
    if (print){
    thrust::copy(values_out.begin(), values_out.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;
    }
  thrust::transform(values_out.begin(), values_out.end(), values_out.begin(), my_sqrt());
  cudaDeviceSynchronize();
 compute_time = dtime_usec(compute_time);

  if (print){
    thrust::copy(values_out.begin(), values_out.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;
    }
  thrust::copy(values_out.begin(), values_out.end(), dist.begin());
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

  thrust::host_vector<float> h_data(data, data + (sizeof(data)/sizeof(float)));
  thrust::host_vector<float> h_centr(centroids, centroids + (sizeof(centroids)/sizeof(float)));
  thrust::host_vector<float> h_dist(k*N);
  eucl_dist_thrust(h_centr, h_data, h_dist, k, n, N, 1);

    
  std::cout<<"\nBye!\n";
  return 0;
}