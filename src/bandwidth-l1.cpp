// example code showing bandwidth of L1 cache and effect of memory alignment

#include <iostream>
#include <omp.h>
#include <sctl.hpp>
using Vec = sctl::Vec<double>;
constexpr int VecLen = Vec::Size();

void profile_write(double* X, long N, long Niter, double val = 3.14) {
  double T = -omp_get_wtime();
  for (long j = 0; j < Niter; j++) {
    Vec v = 3.14;
    #pragma GCC unroll (4)
    for (long i = 0; i < N; i+=VecLen) {
      v.Store(X+i);
    }
  }
  T += omp_get_wtime();
  std::cout<<"Bandwidth = "<< N*Niter*sizeof(double)/T/1e9 <<" GB/s";
  std::cout<<"    cycles/iter = "<< 3.3e9*T/(Niter*N/VecLen) <<"\n";
}

void profile_read(double* X, long N, long Niter) {
  Vec sum[8];
  for (long i = 0; i < 8; i++) sum[i] = 0.0;

  double T = -omp_get_wtime();
  for (long j = 0; j < Niter; j++) {
    for (long i = 0; i < N; i+=VecLen*8) {
      sum[0] = sum[0] + Vec::Load(X+VecLen*0+i);
      sum[1] = sum[1] + Vec::Load(X+VecLen*1+i);
      sum[2] = sum[2] + Vec::Load(X+VecLen*2+i);
      sum[3] = sum[3] + Vec::Load(X+VecLen*3+i);
      sum[4] = sum[4] + Vec::Load(X+VecLen*4+i);
      sum[5] = sum[5] + Vec::Load(X+VecLen*5+i);
      sum[6] = sum[6] + Vec::Load(X+VecLen*6+i);
      sum[7] = sum[7] + Vec::Load(X+VecLen*7+i);
    }
  }
  T += omp_get_wtime();
  std::cout<<"Bandwidth = "<< N*Niter*sizeof(double)/T/1e9 <<" GB/s";
  std::cout<<"    cycles/iter = "<< 3.3e9*T/(Niter*N/VecLen) <<"\n";

  for (long i = 1; i < 8; i++) sum[0] += sum[i];
  if (sum[0][0] < 0) std::cout<<sum[0]<<'\n';
}

void profile_vector_add(double* Y, const double* X, long N, long Niter) { // Y = X + Y
  double T = -omp_get_wtime();
  for (long j = 0; j < Niter; j++) {
    for (long i = 0; i < N; i+=VecLen*2) {
      (Vec::Load(X+VecLen*0+i) + Vec::Load(Y+VecLen*0+i)).Store(Y+VecLen*0+i);
      (Vec::Load(X+VecLen*1+i) + Vec::Load(Y+VecLen*1+i)).Store(Y+VecLen*1+i);
    }
  }
  T += omp_get_wtime();
  std::cout<<"Bandwidth = "<< 3*N*Niter*sizeof(double)/T/1e9 <<" GB/s";
  std::cout<<"    cycles/iter = "<< 3.3e9*T/(Niter*N/VecLen) <<"\n";
}

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cout<<"Usage: ./bandwidth <size-in-bytes> <#-of-iter>\n";
    return 0;
  }

  long N = atol(argv[1]) / sizeof(double);
  long Niter = (argc <= 2 ? std::max<long>(1,1e9/N) : atol(argv[2]));
  std::cout<<"\nSize = "<< N*sizeof(double)<<", Iterations = "<< Niter<<"\n";
  SCTL_ASSERT_MSG(N % 512 == 0, "N must be a multiple of 4192"); // because of vectorizing and loop unrolling

  //double* X = (double*)malloc(N*sizeof(double));
  //double* Y = (double*)malloc(N*sizeof(double));
  double* X = sctl::aligned_new<double>(N);
  double* Y = sctl::aligned_new<double>(N);
  for (long i = 0; i < N; i++) X[i] = Y[i] = i;

  std::cout<<"\n\nWriting to array:\n";
  profile_write(X, N, Niter);

  std::cout<<"\n\nReading from array:\n";
  profile_read(X, N, Niter);

  std::cout<<"\n\nAdding arrays:\n";
  profile_vector_add(Y, X, N, Niter);

  //free(X);
  //free(Y);
  sctl::aligned_delete(X);
  sctl::aligned_delete(Y);
  return 0;
}

