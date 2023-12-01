#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#define threshold 0.0001
// Structure to represent a point in n-dimensional space
struct Point {
  std::vector<double> coordinates;

  Point(std::vector<double> coords) : coordinates(coords) {}
};

// Function to calculate the Euclidean distance between two points
double calculateDistance(const Point& p1, const Point& p2) {
  double distance = 0.0;
  for (size_t i = 0; i < p1.coordinates.size(); ++i) {
    distance += pow((p1.coordinates[i] - p2.coordinates[i]), 2);
  }
  return sqrt(distance);
}

// Function to perform k-means clustering
std::vector<Point> kMeansClustering(const std::vector<Point>& points, int k,
                                    int maxIterations) {
  if (points.empty() || k <= 0 || k > points.size() || maxIterations <= 0) {
    std::cerr << "Invalid input parameters" << std::endl;
    return {};
  }

  std::vector<Point> centroids;

  // Initialize centroids by choosing the first k points from the dataset
  for (int i = 0; i < k; ++i) {
    centroids.push_back(points[i]);
  }

  std::vector<int> clusters(points.size());  // To store cluster assignment for each point

  double delta = std::numeric_limits<double>::max();
  int iterations = 0;
  int N = points.size();
  while (delta / N > threshold && iterations < maxIterations) {
    delta = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
      double minDist = std::numeric_limits<double>::max();
      int clusterIndex = -1;
      for (int j = 0; j < k; ++j) {
        double dist = calculateDistance(points[i], centroids[j]);
        if (dist < minDist) {
          minDist = dist;
          clusterIndex = j;
        }
      }
      if (clusters[i] != clusterIndex) {
        ++delta;
        clusters[i] = clusterIndex;
      }
    }

    // Update centroids by calculating the mean of points in each cluster
    // std::vector<Point> newCentroids(k, Point(points[0].coordinates.size(),
    // std::vector<double>(0)));

    std::vector<Point> newCentroids;
    for (int i = 0; i < k; ++i) {
      std::vector<double> initialCoordinates(points[0].coordinates.size(), 0.0);
      newCentroids.emplace_back(initialCoordinates);
    }

    std::vector<int> clusterCounts(k, 0);

    for (size_t i = 0; i < points.size(); ++i) {
      int clusterIndex = clusters[i];
      clusterCounts[clusterIndex]++;
      for (size_t j = 0; j < points[i].coordinates.size(); ++j) {
        newCentroids[clusterIndex].coordinates[j] += points[i].coordinates[j];
      }
    }

    for (int i = 0; i < k; ++i) {
      for (size_t j = 0; j < newCentroids[i].coordinates.size(); ++j) {
        newCentroids[i].coordinates[j] /= clusterCounts[i];
      }
    }

    centroids = newCentroids;
  }

  return centroids;
}

// Example usage
int main() {
  std::vector<Point> points = {
      Point({2.0, 3.0}), Point({5.0, 6.0}), Point({8.0, 7.0}),
      // Add more points as needed...
  };

  int k = 2;  // Number of clusters
  int maxIterations = 100;

  std::vector<Point> centroids = kMeansClustering(points, k, maxIterations);

  // Display centroids
  for (size_t i = 0; i < centroids.size(); ++i) {
    std::cout << "Centroid " << i + 1 << ": ";
    for (const auto& coordinate : centroids[i].coordinates) {
      std::cout << coordinate << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
