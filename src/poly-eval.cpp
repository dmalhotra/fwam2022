// example code showing effect of pipelining in evaluating polynomial

#include <iostream>
#include <sctl.hpp>
#include <omp.h>

#define CPU_clockrate 3.3 // GHz
constexpr int VecLen = sctl::DefaultVecLen<double>();

template <class Type> void test_polynomial() {
  Type a,b,c,d,e,f,g,h; // coefficients
  a =  2.3515e-07;
  b =  9.8697e-04;
  c = -1.8656e-02;
  d =  1.0716e-01;
  e = -1.1821e-01;
  f = -3.9467e-01;
  g = -3.8480e-02;
  h =  1.0033e+00;
  Type x = drand48();

  std::cout<<"\n\nEvaluating polynomials using Horner's rule:\n";
  double T = -omp_get_wtime();
  for (long i = 0; i < 1000000000L; i++) {
    x = (((((a*x+b)*x+c)*x+d)*x+e)*x+f*x+g)*x+h;
  }
  T += omp_get_wtime();
  std::cout<<"T = "<< T <<'\n';
  std::cout<<"cycles/iter = "<< CPU_clockrate*T <<'\n';

  std::cout<<"\n\nEvaluating polynomials using Estrin's method:\n";
  T = -omp_get_wtime();
  for (long i = 0; i < 1000000000L; i++) {
    Type x2 = x * x;
    Type x4 = x2 * x2;
    x = ((a*x+b)*x2+(c*x+d))*x4+(e*x+f)*x2+(g*x+h);
  }
  T += omp_get_wtime();
  std::cout<<"T = "<< T <<'\n';
  std::cout<<"cycles/iter = "<< CPU_clockrate*T <<'\n';

  std::cout<<"\n\nEvaluating polynomials using Estrin's method (unrolled):\n";
  T = -omp_get_wtime();
  for (long i = 0; i < 1000000000L; i++) {
    Type x2 = x * x;
    Type x4 = x2 * x2;
    Type u = a * x + b;
    Type v = c * x + d;
    Type w = e * x + f;
    Type p = g * x + h;
    Type q = u * x2 + v;
    Type r = w * x2 + p;
    x = q * x4 + r;
  }
  T += omp_get_wtime();
  std::cout<<"T = "<< T <<'\n';
  std::cout<<"cycles/iter = "<< CPU_clockrate*T <<'\n';

  std::cout<<"Result = "<<x<<"\n\n\n";
}

int main(int argc, char** argv) {

  std::cout<<"\n\nCPU clockrate = "<<CPU_clockrate<<"\n";

  test_polynomial<double>(); // scalar

  //test_polynomial<sctl::Vec<double,VecLen>>(); // vectorized

  return 0;
}

