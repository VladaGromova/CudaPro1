#include <iostream>
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

#define MAX_DATA 100000000
#define MAX_CENT 5000
#define TOL 0.001

unsigned long long dtime_usec(unsigned long long prev){
#define USECPSEC 1000000ULL
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}

unsigned verify(float *d1, float *d2, int len){
  unsigned pass = 1;
  for (int i = 0; i < len; i++)
    if (fabsf(d1[i] - d2[i]) > TOL){
      std::cout << "mismatch at:  " << i << " val1: " << d1[i] << " val2: " << d2[i] << std::endl;
      pass = 0;
      break;}
  return pass;
}
void eucl_dist_cpu(const float *centroids, const float *data, float *rdist, int num_centroids, int dim, int num_data, int print){

  int out_idx = 0;
  float dist, dist_sqrt;
  for(int i = 0; i < num_centroids; i++)
    for(int j = 0; j < num_data; j++)
    {
        float dist_sum = 0.0;
        for(int k = 0; k < dim; k++)
        {
            dist = centroids[i * dim + k] - data[j * dim + k];
            dist_sum += dist * dist;
        }
        dist_sqrt = sqrt(dist_sum);
        // do something with the distance
        rdist[out_idx++] = dist_sqrt;
        if (print) std::cout << dist_sqrt << ", ";

    }
    if (print) std::cout << std::endl;
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


unsigned long long eucl_dist_thrust(thrust::host_vector<float> &centroids, thrust::host_vector<float> &data, thrust::host_vector<float> &dist, int num_centroids, int dim, int num_data, int print){

  thrust::device_vector<float> d_data = data;
  thrust::device_vector<float> d_centr = centroids;
  thrust::device_vector<float> values_out(num_centroids*num_data);

  unsigned long long compute_time = dtime_usec(0);
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), dkeygen(dim, num_data)), thrust::make_transform_iterator(thrust::make_counting_iterator<int>(dim*num_data*num_centroids), dkeygen(dim, num_data)),thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_centr.begin(), thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), c_idx(dim, num_data))), thrust::make_permutation_iterator(d_data.begin(), thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), d_idx(dim, num_data))))), my_dist()), thrust::make_discard_iterator(), values_out.begin());
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


int main(int argc, char *argv[]){

  int dim = 8;
  int num_centroids = 2;
  float centroids[] = {
    0.223, 0.002, 0.223, 0.412, 0.334, 0.532, 0.244, 0.612,
    0.742, 0.812, 0.817, 0.353, 0.325, 0.452, 0.837, 0.441
  };
  int num_data = 8;
  float data[] = {
    0.314, 0.504, 0.030, 0.215, 0.647, 0.045, 0.443, 0.325,
    0.731, 0.354, 0.696, 0.604, 0.954, 0.673, 0.625, 0.744,
    0.615, 0.936, 0.045, 0.779, 0.169, 0.589, 0.303, 0.869,
    0.275, 0.406, 0.003, 0.763, 0.471, 0.748, 0.230, 0.769,
    0.903, 0.489, 0.135, 0.599, 0.094, 0.088, 0.272, 0.719,
    0.112, 0.448, 0.809, 0.157, 0.227, 0.978, 0.747, 0.530,
    0.908, 0.121, 0.321, 0.911, 0.884, 0.792, 0.658, 0.114,
    0.721, 0.555, 0.979, 0.412, 0.007, 0.501, 0.844, 0.234
  };
  std::cout << "cpu results: " << std::endl;
  float dist[num_data*num_centroids];
  eucl_dist_cpu(centroids, data, dist, num_centroids, dim, num_data, 1);

  thrust::host_vector<float> h_data(data, data + (sizeof(data)/sizeof(float)));
  thrust::host_vector<float> h_centr(centroids, centroids + (sizeof(centroids)/sizeof(float)));
  thrust::host_vector<float> h_dist(num_centroids*num_data);

  std::cout << "gpu results: " << std::endl;
  eucl_dist_thrust(h_centr, h_data, h_dist, num_centroids, dim, num_data, 1);

  float *data2, *centroids2, *dist2;
  num_centroids = 10;
  num_data = 1000000;

  if (argc > 2) {
    num_centroids = atoi(argv[1]);
    num_data = atoi(argv[2]);
    if ((num_centroids < 1) || (num_centroids > MAX_CENT)) {std::cout << "Num centroids out of range" << std::endl; return 1;}
    if ((num_data < 1) || (num_data > MAX_DATA)) {std::cout << "Num data out of range" << std::endl; return 1;}
    if (num_data * dim * num_centroids > 2000000000) {std::cout << "data set out of range" << std::endl; return 1;}}
  std::cout << "Num Data: " << num_data << std::endl;
  std::cout << "Num Cent: " << num_centroids << std::endl;
  std::cout << "result size: " << ((num_data*num_centroids*4)/1048576) << " Mbytes" << std::endl;
  data2 = new float[dim*num_data];
  centroids2 = new float[dim*num_centroids];
  dist2 = new float[num_data*num_centroids];
  for (int i = 0; i < dim*num_data; i++) data2[i] = rand()/(float)RAND_MAX;
  for (int i = 0; i < dim*num_centroids; i++) centroids2[i] = rand()/(float)RAND_MAX;
  unsigned long long dtime = dtime_usec(0);
  eucl_dist_cpu(centroids2, data2, dist2, num_centroids, dim, num_data, 0);
  dtime = dtime_usec(dtime);
  std::cout << "cpu time: " << dtime/(float)USECPSEC << "s" << std::endl;
  thrust::host_vector<float> h_data2(data2, data2 + (dim*num_data));
  thrust::host_vector<float> h_centr2(centroids2, centroids2 + (dim*num_centroids));
  thrust::host_vector<float> h_dist2(num_data*num_centroids);
  dtime = dtime_usec(0);
  unsigned long long ctime = eucl_dist_thrust(h_centr2, h_data2, h_dist2, num_centroids, dim, num_data, 0);
  dtime = dtime_usec(dtime);
  std::cout << "gpu total time: " << dtime/(float)USECPSEC << "s, gpu compute time: " << ctime/(float)USECPSEC << "s" << std::endl;
  if (!verify(dist2, &(h_dist2[0]), num_data*num_centroids)) {std::cout << "Verification failure." << std::endl; return 1;}
  std::cout << "Success!" << std::endl;

  return 0;

}