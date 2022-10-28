// example code showing blocking of GEMM to optimize memory access
//
// Cache profiling using perf:
// perf stat -e L1-dcache-load-misses -e L1-dcache-loads -e l2_rqsts.miss -e l2_rqsts.references -e LLC-load-misses -e LLC-loads ./$@

#include <iostream>
#include <omp.h>
#include <sctl.hpp>

constexpr int VecLen = sctl::DefaultVecLen<double>();

void GEMM_naive(int M, int N, int K, double* A, int LDA, double* B, int LDB, double* C, int LDC) {
  for (int j = 0; j < N; j++)
    for (int k = 0; k < K; k++)
      for (int i = 0; i < M; i++)
        C[i+j*LDC] += A[i+k*LDA] * B[k+j*LDB];
}

template <int M, int N, int K>
void GEMM_ker_vec_unrolled(double* A, int LDA, double* B, int LDB, double* C, int LDC) {
  using Vec = sctl::Vec<double,M>;

  Vec C_vec[N];
  #pragma GCC unroll (10)
  for (int j = 0; j < N; j++)
    C_vec[j] = Vec::Load(C+j*LDC);

  #pragma GCC unroll (40)
  for (int k = 0; k < K; k++) {
    const Vec A_vec = Vec::Load(A+k*LDA);
    double* B_ = B + k;
    #pragma GCC unroll (10)
    for (int j = 0; j < N; j++) {
      C_vec[j] = A_vec * B_[j*LDB] + C_vec[j];
    }
  }

  #pragma GCC unroll (10)
  for (int j = 0; j < N; j++)
    C_vec[j].Store(C+j*LDC);
}

template <int M, int N, int K>
void GEMM_blocked(double* A, int LDA, double* B, int LDB, double* C, int LDC) {
  if (M == sctl::DefaultVecLen<double>()) {
    GEMM_ker_vec_unrolled<M,N,K>(A,LDA, B,LDB, C,LDC);
    return;
  }

  for (int j = 0; j < N; j++)
    for (int k = 0; k < K; k++)
      for (int i = 0; i < M; i++)
        C[i+j*LDC] += A[i+k*LDA] * B[k+j*LDB];
}

template <int M, int N, int K, int Mb, int Nb, int Kb, int... NN>
void GEMM_blocked(double* A, int LDA, double* B, int LDB, double* C, int LDC) {
  static_assert(M % Mb == 0);
  static_assert(N % Nb == 0);
  static_assert(K % Kb == 0);
  for (int j = 0; j < N; j+=Nb)
    for (int i = 0; i < M; i+=Mb)
      for (int k = 0; k < K; k+=Kb)
        GEMM_blocked<Mb,Nb,Kb, NN...>(A+i+k*LDA,LDA, B+k+j*LDB,LDB, C+i+j*LDC,LDC);
}

int main(int argc, char** argv) {
  constexpr long M = 2000, N = 2000, K = 2000, iter = 10;
  double* C_ref = new double[M*N];
  double* C = new double[M*N];
  double* A = new double[M*K];
  double* B = new double[K*N];
  double T = 0;

  for (long i = 0; i < M*N; i++) C[i] = 0;
  for (long i = 0; i < M*N; i++) C_ref[i] = 0;
  for (long i = 0; i < M*K; i++) A[i] = drand48();
  for (long i = 0; i < K*N; i++) B[i] = drand48();

  T = -omp_get_wtime();
  for (long i = 0; i < iter; i++)
    //GEMM_naive(M,N,K, A,M, B,K, C,M);
    GEMM_blocked<M,N,K, 200,200,200, 40,40,40, VecLen,10,40>(A,M, B,K, C,M);
  T += omp_get_wtime();
  std::cout<<"M = "<<M<<"  N = "<<N<<"  K = "<<K<<'\n';
  std::cout<<"T = "<<T<<"    GFLOPS = "<<2*M*N*K*iter/T/1e9<<'\n';

  if (0) { // verify result
    T = -omp_get_wtime();
    for (long i = 0; i < iter; i++)
      GEMM_naive(M,N,K, A,M, B,K, C_ref,M);
    T += omp_get_wtime();
    std::cout<<"T = "<<T<<"    GFLOPS = "<<2*M*N*K*iter/T/1e9<<'\n';

    double max_err = 0, max_val = 0;
    for (long i = 0; i < M*N; i++) {
      max_err = std::max(max_err, fabs(C[i]-C_ref[i]));
      max_val = std::max(max_val, fabs(C_ref[i]));
    }
    std::cout<<"Error = "<<max_err/max_val<<'\n';
  }

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C_ref;
  return 0;
}
