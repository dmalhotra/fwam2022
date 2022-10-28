// example code showing cost of memory initialization (first-touch) and NUMA
//
// OpenMP thread pinning for NUMA
// export OMP_PLACES=cores
// export OMP_PROC_BIND=spread

#include <iostream>
#include <omp.h>
#include <sctl.hpp>
using Vec = sctl::Vec<double>;
constexpr int VecLen = Vec::Size();

// Benchmark to show cost of memory allocations
void benchmark_memory_init() {
  long N = 1e9; // 8 GB
  double T;

  // Allocate memory
  T = -omp_get_wtime();
  double* X = (double*)malloc(N*sizeof(double));
  std::cout<<"Array alloc time = "<<T+omp_get_wtime()<<'\n';

  // Initialize array
  T = -omp_get_wtime();
  for (long i = 0; i < N; i++) X[i] = i;
  std::cout<<"Array init time  = "<<T+omp_get_wtime()<<'\n';

  // Write to array
  T = -omp_get_wtime();
  for (long i = 0; i < N; i++) X[i] = 2*i;
  std::cout<<"Array write time = "<<T+omp_get_wtime()<<'\n';

  // Free memory
  T = -omp_get_wtime();
  free(X);
  std::cout<<"Array free time  = "<<T+omp_get_wtime()<<'\n';
}

// Benchmark to show effect of NUMA
void benchmark_numa(bool numa_aware) {
  long N = 1e9; // 8 BG
  double T;

  // Allocate memory
  double* X = sctl::aligned_new<double>(N);
  double* Y = sctl::aligned_new<double>(N);

  // Initialize X, Y : this is when memory pages are assigned to each NUMA node
  if (numa_aware) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) X[i] = Y[i] = i;
  } else {
    for (long i = 0; i < N; i++) X[i] = Y[i] = i;
  }

  // Write to array
  T = -omp_get_wtime();
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) X[i] = 3.14;
  T += omp_get_wtime();
  std::cout<<"Write Bandwidth   = "<< N*sizeof(double)/T/1e9 <<" GB/s\n";

  // Read from array
  double sum = 0;
  T = -omp_get_wtime();
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += X[i];
  T += omp_get_wtime();
  std::cout<<"Read Bandwidth    = "<< N*sizeof(double)/T/1e9 <<" GB/s\n";
  if (sum < 0) std::cout<<sum<<'\n';

  // Adding arrays: 2-reads, 1-write
  T = -omp_get_wtime();
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) Y[i] += X[i];
  T += omp_get_wtime();
  std::cout<<"Vec-Add Bandwidth = "<< 3*N*sizeof(double)/T/1e9 <<" GB/s\n";

  sctl::aligned_delete(X);
  sctl::aligned_delete(Y);
}

int main(int argc, char** argv) {

  std::cout<<"\nBenchmarking memory initialization cost:\n";
  benchmark_memory_init();

  std::cout<<"\n\nBenchmarking main memory without parallel initialization (NUMA unaware):\n";
  benchmark_numa(false);

  std::cout<<"\n\nBenchmarking main memory with parallel initialization (NUMA aware):\n";
  benchmark_numa(true);

  return 0;
}

