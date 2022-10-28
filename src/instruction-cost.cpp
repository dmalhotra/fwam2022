// example codes showing instruction latency and throughput

#include <iostream>
#include <sctl.hpp>
#include <omp.h>

#define CPU_clockrate 3.3 // GHz
constexpr int VecLen = sctl::DefaultVecLen<double>();

template <class Type, int K> void test_add() { // add K elements of Type
  Type x[K], one = 1.0;
  for (long k = 0; k < K; k++)
    x[k] = 3.14 + k; // initialize x[k]

  double T = -omp_get_wtime();
  for (long i = 0; i < 1000000000L; i++)
    for (long k = 0; k < K; k++)
      x[k] = one + x[k];
  T += omp_get_wtime();
  std::cout<<"T = "<< T <<'\n';
  std::cout<<"cycles/iter = "<< CPU_clockrate*T <<'\n';

  // print the result otherwise the
  // compiler skips everything
  Type sum = 0.;
  for (long k = 0; k < K; k++) sum += x[k];
  std::cout<<"Result = "<<sum<<'\n';
}

template <class Type, int K> void test_division() { // divide K elements of Type
  Type x[K], one = 1.0;
  for (long k = 0; k < K; k++)
    x[k] = 3.14 + k; // initialize x[k]

  double T = -omp_get_wtime();
  for (long i = 0; i < 1000000000L; i++)
    for (long k = 0; k < K; k++)
      x[k] = one / x[k];
  T += omp_get_wtime();
  std::cout<<"T = "<< T <<'\n';
  std::cout<<"cycles/iter = "<< CPU_clockrate*T <<'\n';

  // print the result otherwise the
  // compiler skips everything
  Type sum = 0.;
  for (long k = 0; k < K; k++) sum += x[k];
  std::cout<<"Result = "<<sum<<'\n';
}

int main(int argc, char** argv) {

  std::cout<<"\n\nCPU clockrate = "<<CPU_clockrate<<"\n";

  std::cout<<"\n\nAdding one doubles at a time:\n";
  test_add<double, 1>();

  std::cout<<"\n\nAdding 32 doubles at a time:\n";
  test_add<double, 32>();

  std::cout<<"\n\nAdding 8 Vec<doubles,"<<VecLen<<"> at a time:\n";
  test_add<sctl::Vec<double,VecLen>, 8>();

  std::cout<<"\n\nDividing 8 Vec<doubles,"<<VecLen<<"> at a time:\n";
  test_division<sctl::Vec<double,8>,VecLen>();

  return 0;
}




