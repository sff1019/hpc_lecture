#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

const int N = 8;

float _mm256_reduce(__m256 vec) {
  __m256 permutevec = _mm256_permute2f128_ps(vec, vec, 1);
  permutevec = _mm256_add_ps(permutevec, vec);
  permutevec = _mm256_hadd_ps(permutevec, permutevec);
  permutevec = _mm256_hadd_ps(permutevec, permutevec);

  float reducedvec[N];
  _mm256_store_ps(reducedvec, permutevec);

  return reducedvec[0];
}

int main() {
  float x[N], y[N], m[N], fx[N], fy[N];

  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  for (int i = 0; i < N; i++) {
    __m256 xvec = _mm256_load_ps(x);
    __m256 yvec = _mm256_load_ps(y);
    __m256 mvec = _mm256_load_ps(m);

    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);

    __m256 mask = _mm256_cmp_ps(xvec, xivec, _CMP_NEQ_OQ);
    __m256 zerovec = _mm256_setzero_ps();

    xvec = _mm256_blendv_ps(zerovec, xvec, mask);
    yvec = _mm256_blendv_ps(zerovec, yvec, mask);
    mvec = _mm256_blendv_ps(zerovec, mvec, mask);

    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);

    __m256 rvec = _mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec));
    rvec = _mm256_rsqrt_ps(rvec);

    __m256 rxmvec = _mm256_mul_ps(rxvec, mvec);
    rxmvec = _mm256_mul_ps(rxmvec,rvec);
    rxmvec = _mm256_mul_ps(rxmvec,rvec);
    rxmvec = _mm256_mul_ps(rxmvec,rvec);

    __m256 rymvec = _mm256_mul_ps(ryvec, mvec);
    rymvec = _mm256_mul_ps(rymvec,rvec);
    rymvec = _mm256_mul_ps(rymvec,rvec);
    rymvec = _mm256_mul_ps(rymvec,rvec);

    fx[i] -= _mm256_reduce(rxmvec);
    fy[i] -= _mm256_reduce(rymvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }

  // for(int i=0; i<N; i++) {
  //   for(int j=0; j<N; j++) {
  //     if(i != j) {
  //       double rx = x[i] - x[j];
  //       double ry = y[i] - y[j];
  //       double r = std::sqrt(rx * rx + ry * ry);
  //       fx[i] -= rx * m[j] / (r * r * r);
  //       fy[i] -= ry * m[j] / (r * r * r);
  //     }
  //   }
  //   printf("%d %g %g\n",i,fx[i],fy[i]);
  // }
}
