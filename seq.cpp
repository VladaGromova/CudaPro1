#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
#define threshold 0.0000001
#define MAX_IT 100
using namespace std::chrono;

struct Point {
  std::vector<double> coordinates;
  Point(std::vector<double> coords) : coordinates(coords) {}
};

double calculateDistance(const Point& p1, const Point& p2) {
  double distance = 0.0;
  for (size_t i = 0; i < p1.coordinates.size(); ++i) {
    distance += pow((p1.coordinates[i] - p2.coordinates[i]), 2);
  }
  return sqrt(distance);
}

std::vector<Point> kMeansClustering(const std::vector<Point>& points, int k, std::vector<int>& clusterIndexes) {
  if (points.empty() || k <= 0 || k > points.size() || MAX_IT <= 0) {
    std::cerr << "Invalid input parameters" << std::endl;
    return {};
  }
  std::vector<Point> centroids;
  for (int i = 0; i < k; ++i) {
    centroids.push_back(points[i]);
  }

  std::vector<int> clusters(points.size());

  double delta = std::numeric_limits<double>::max();
  int iterations = 0;
  int N = points.size();
  while (delta / N > threshold && iterations < MAX_IT) {
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
    ++iterations;
    centroids = newCentroids;
  }
  std::cout << "\nIterations: " << iterations << '\n';
  clusterIndexes = clusters;
  return centroids;
}

std::vector<Point> readFile(std::istream& inputFile, int* N, int* n, int* k) {
  //std::ifstream inputFile(FILENAME);
  std::string inputString;
  std::vector<double> pointsFromString = {};
  getline(inputFile, inputString);
  *N = stoi(inputString);
  getline(inputFile, inputString);
  *n = stoi(inputString);
  getline(inputFile, inputString);
  *k = stoi(inputString);

  std::string word;

  std::vector<Point> points;
  while (getline(inputFile, inputString)) {
    std::istringstream iss(inputString);
    std::vector<double> coords;
    double val;
    while (iss >> val) {
      coords.push_back(val);
    }

    Point newPoint(coords);
    points.push_back(newPoint);
  }
  //inputFile.close();
  return points;
}

void printPoint(Point p) {
  for (int i = 0; i < p.coordinates.size(); i++) {
    std::cout << p.coordinates[i] << ' ';
  }
}

void printPoints(std::vector<Point> points) {
  for (int i = 0; i < points.size(); i++) {
    printPoint(points[i]);
    std::cout << '\n';
  }
}

void writeDataToFile(std::vector<Point> points, std::vector<int> clusters, int N, int n) {
  std::ofstream outputFile;
  outputFile.open("out_seq.txt");
  if (outputFile.is_open()) {
    for (int i = 0; i < points.size(); i++) {
      for (int j = 0; j < points[i].coordinates.size(); j++) {
        outputFile << points[i].coordinates[j] << ' ';
      }
      outputFile << clusters[i];
      outputFile << '\n';
    }
    outputFile.close();
    std::cout << "Data written successfully \n" << std::endl;
  } else {
    std::cout << "Unable to open the file \n" << std::endl;
  }
}

int main(int argc, char** argv) {
  auto start = high_resolution_clock::now();
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


  int N, n, k;
  std::vector<Point> points = readFile(inputFile, &N, &n, &k);
  inputFile.close();

  std::vector<int> clusterIndexes(N);
  std::vector<Point> centroids = kMeansClustering(points, k, clusterIndexes);
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(stop - start);

  std::cout << "Time taken by function: " << duration.count()
            << " ms\n";
  
  writeDataToFile(points,clusterIndexes ,N, n);

  return 0;
}
