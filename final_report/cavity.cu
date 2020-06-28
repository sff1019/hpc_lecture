#include <cstdio>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;

int nx = 41;
int ny = 41;
int grid_size = nx * ny;
int SIZE = grid_size * sizeof(float);
int nt = 700;
int nit = 50;
float c = 1.0;
float dx = 2.0 / (nx - 1);
float dy = 2.0 / (ny - 1);

int rho = 1.0;
float nu = 0.1;
float dt = 0.001;

// CUDA configs
int n_threads = 1024;  // per block
int n_blocks = (nx * ny + n_threads - 1) / n_threads;

__host__ void writeFile(float* u, float* v, float* p) {
  ofstream fs("cavity_cu_results.txt");

  // u
  fs << "u ";
  for (int i = 0; i < grid_size; i++) fs << u[i] << " ";
  fs << "\n";
  fs << "v ";
  for (int i = 0; i < grid_size; i++) fs << v[i] << " ";
  fs << "\n";
  fs << "p ";
  for (int i = 0; i < grid_size; i++) fs << p[i] << " ";
  fs << "\n";

  fs.close();
}

__global__ void build_up_b(float *u, float *v, float *b, int nx, int ny, float dx, float dy, float dt, float rho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nx) return;
  else if (i % nx == 0) return;
  else if (i % nx == nx - 1) return;
  else if (i > (ny - 1) * nx - 1) return;

  b[i] = (rho * ( 1.0 / dt *
        ((u[i+1] - u[i-1]) /
         (2.0 * dx) + (v[i+nx] - v[i-nx]) / (2.0 * dy)) -
        ((u[i+1] - u[i-1]) / (2.0 * dx)) * ((u[i+1] - u[i-1]) / (2.0 * dx)) -
        2.0 * ((u[i+nx] - u[i-nx]) / (2.0 * dy) *
          (v[i+1] - v[i-1]) / (2.0 * dx)) -
        ((v[i+nx] - v[i-nx]) / (2.0 * dy)) * ((v[i+nx] - v[i-nx]) / (2.0 * dy))));
}

__global__ void pressure_poisson(float *p, float *pn, float *b, int nx, int ny, float dx, float dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i >= nx) && (i % nx != 0) && (i % nx != nx - 1) && (i <= nx * (ny - 1) - 1)) {
    p[i] = (((pn[i+1] + pn[i-1]) * pow(dy, 2) +
          (pn[i+nx] + pn[i-nx]) * pow(dx, 2)) /
        (2 * (pow(dx, 2) + pow(dy, 2))) -
        pow(dx, 2) * pow(dy, 2) / (2 * (pow(dx, 2) + pow(dy, 2))) *
        b[i]);
  }

  if (i >= nx * ny) return;
  if (i % nx == nx - 1) p[i] = p[i-1];
  else if (i % nx == 0) p[i] = p[i+1];
  else if (i < nx) p[i] = p[i+nx];
  else if (i >= nx*(ny - 1)) p[i] = 0.0;
}

__global__ void cavity_flow_u_update(float *u, float *un, float *vn, float *p, int nx, int ny, float nu, float dx, float dy, float dt, float rho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i >= nx) && (i % nx != nx - 1) && (i % nx != 0) && (i < nx * (ny - 1))) {
    u[i] = (un[i] -
        un[i] * dt / dx *
        (un[i] - un[i-1]) -
        vn[i] * dt / dy *
        (un[i] - un[i-nx]) -
        dt/ (2.0 * rho * dx) * (p[i+1] - p[i-1]) +
        nu * (dt / pow(dx, 2.0) *
          (un[i+1] - 2.0 * un[i] + un[i-1]) +
          dt / pow(dy, 2.0) *
          (un[i+nx] - 2.0 * un[i] + un[i-nx])));
  }

  if(i >= nx*ny) return;
  if(i%nx == nx-1) u[i] = 0.0;
  else if(i < nx) u[i] = 0.0;
  else if(i%nx == 0) u[i] = 0.0;
  else if(i >= nx*(ny-1)) u[i] = 1.0;

}

__global__ void cavity_flow_v_update(float *v, float *vn, float *un, float *p, int nx, int ny, float nu, float dx, float dy, float dt, float rho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i >= nx) && (i % nx != nx - 1) && (i % nx != 0) && (i < nx * (ny - 1))) {
    v[i] = (vn[i] -
        vn[i] * dt / dx *
        (vn[i] - vn[i-1]) -
        vn[i] * dt / dy *
        (vn[i] - vn[i-nx]) -
        dt / (2 * rho * dy) * (p[i+nx] - p[i-nx]) +
        nu * (dt / pow(dx, 2) *
          (vn[i+1] - 2 * vn[i] + vn[i-1]) +
          dt / pow(dy, 2) *
          (vn[i+nx] - 2.0 * vn[i] + vn[i-nx])));
  }

  if(i >= nx*ny) return;
  if(i%nx == nx-1) v[i] = 0.0;
  else if(i < nx) v[i] = 0.0;
  else if(i%nx == 0) v[i] = 0.0;
  else if(i >= nx*(ny-1)) v[i] = 0.0;

}

void cavity_flow(float *u, float *v, float *p) {
  float *un, *vn, *pn, *b;
  cudaMallocManaged(&un, SIZE);
  cudaMallocManaged(&vn, SIZE);
  cudaMallocManaged(&pn, SIZE);
  cudaMallocManaged(&b, SIZE);

  cudaMemset(un, 0.0, SIZE);
  cudaMemset(vn, 0.0, SIZE);
  cudaMemset(pn, 0.0, SIZE);
  cudaMemset(b, 0.0, SIZE);

  for (int n = 0; n < nt; n++) {
    cudaMemcpy(un, u, SIZE, cudaMemcpyDeviceToDevice);
    cudaMemcpy(vn, v, SIZE, cudaMemcpyDeviceToDevice);

    build_up_b<<<n_blocks, n_threads>>>(u, v, b, nx, ny, dx, dy, dt, rho);
    cudaDeviceSynchronize();
    for (int q = 0; q < nit; q++) {
      cudaMemcpy(pn, p, SIZE, cudaMemcpyDeviceToDevice);
      pressure_poisson<<<n_blocks, n_threads>>>(p, pn, b, nx, ny, dx, dy);
      cudaDeviceSynchronize();
    }
    cavity_flow_u_update<<<n_blocks, n_threads>>>(u, un, vn, p, nx, ny, nu, dx, dy, dt, rho);
    cavity_flow_v_update<<<n_blocks, n_threads>>>(v, vn, un, p, nx, ny, nu, dx, dy, dt, rho);
    cudaDeviceSynchronize();
  }

  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);
  cudaFree(b);
}

int main() {
  float *u, *v, *p;
  cudaMallocManaged(&u, SIZE);
  cudaMallocManaged(&v, SIZE);
  cudaMallocManaged(&p, SIZE);

  cudaMemset(u, 0.0, SIZE);
  cudaMemset(v, 0.0, SIZE);
  cudaMemset(p, 0.0, SIZE);

  cavity_flow(u, v, p);

  writeFile(u, v, p);

  cudaFree(u);
  cudaFree(v);
  cudaFree(p);

  return 0;
}
