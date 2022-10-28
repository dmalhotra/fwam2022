// example code showing optimization of GEMM micro-kernel

#include <iostream>
#include <omp.h>
#include <sctl.hpp>

constexpr int VecLen = sctl::DefaultVecLen<double>();

template <int M, int N, int K>
void GEMM_ker_naive(double* C, double* A, double* B) {
  for (int k = 0; k < K; k++)
    for (int j = 0; j < N; j++)
      for (int i = 0; i < M; i++)
        C[i+j*M] += A[i+k*M] * B[k+K*j];
}

template <int M, int N, int K>
void GEMM_ker_vec(double* C, double* A, double* B) {
  using Vec = sctl::Vec<double,M>;

  Vec C_vec[N];
  for (int j = 0; j < N; j++)
    C_vec[j] = Vec::Load(C+j*M);

  for (int k = 0; k < K; k++) {
    const Vec A_vec = Vec::Load(A+k*M);
    double* B_ = B + k;
    for (int j = 0; j < N; j++) {
      C_vec[j] = A_vec * B_[K*j] + C_vec[j];
    }
  }

  for (int j = 0; j < N; j++)
    C_vec[j].Store(C+j*M);
}

template <int M, int N, int K>
void GEMM_ker_vec_unrolled(double* C, double* A, double* B) {
  using Vec = sctl::Vec<double,M>;

  Vec C_vec[N];
  #pragma GCC unroll (10)
  for (int j = 0; j < N; j++)
    C_vec[j] = Vec::Load(C+j*M);

  #pragma GCC unroll (40)
  for (int k = 0; k < K; k++) {
    const Vec A_vec = Vec::Load(A+k*M);
    double* B_ = B + k;
    #pragma GCC unroll (10)
    for (int j = 0; j < N; j++) {
      C_vec[j] = A_vec * B_[j*K] + C_vec[j];
    }
  }

  #pragma GCC unroll (10)
  for (int j = 0; j < N; j++)
    C_vec[j].Store(C+j*M);
}

int main(int argc, char** argv) {
  long L = 1e6;
  constexpr int M = VecLen, N = 10, K = 40;
  double* C = new double[M*N];
  double* A = new double[M*K];
  double* B = new double[K*N];
  for (long i = 0; i < M*N; i++) C[i] = 0;
  for (long i = 0; i < M*K; i++) A[i] = drand48();
  for (long i = 0; i < K*N; i++) B[i] = drand48();

  std::cout<<"M = "<<M<<", N = "<<N<<", K = "<<K<<"\n\n";

  std::cout<<"GEMM (naive)\n";
  double T = -omp_get_wtime();
  for(long i = 0; i < L; i++) GEMM_ker_naive<M,N,K>(C, A, B);
  T += omp_get_wtime();
  std::cout<<"FLOP rate = "<< 2*M*N*K*L/T/1e9 <<" GFLOP/s\n\n\n";

  std::cout<<"GEMM (vectorized)\n";
  T = -omp_get_wtime();
  for(long i = 0; i < L; i++) GEMM_ker_vec<M,N,K>(C, A, B);
  std::cout<<"FLOP rate = "<< 2*M*N*K*L/(T+omp_get_wtime())/1e9 <<" GFLOP/s\n\n\n";

  std::cout<<"GEMM (vectorized & unrolled)\n";
  T = -omp_get_wtime();
  for(long i = 0; i < L; i++) GEMM_ker_vec_unrolled<M,N,K>(C, A, B);
  std::cout<<"FLOP rate = "<< 2*M*N*K*L/(T+omp_get_wtime())/1e9 <<" GFLOP/s\n\n\n";

  double sum = 0;
  for (long i = 0; i < M*N; i++) sum += C[i];
  std::cout<<"result = "<<sum<<'\n';

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
