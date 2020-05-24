#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucketSort(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  atomicAdd(&bucket[key[i]], 1);
  __syncthreads();

  for (int j=0, k=0; j <= i; k++) {
    key[i] = k;
    j += bucket[k];
  }
}

int main() {
  const int n = 50;
  const int m = 64;
  int range = 5;

  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));

  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }

  bucketSort<<<(n+m-1)/m, m>>>(key, bucket, n);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
