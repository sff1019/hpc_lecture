#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

int nx = 41;
int ny = 41;
int MAT_SIZE = nx * ny;
int nt = 700;
int nit = 50;
float c = 1.0;
float dx = 2.0 / (nx - 1);
float dy = 2.0 / (ny - 1);

int rho = 1.0;
float nu = 0.1;
float dt = 0.001;

void writeFile(float* u, float* v, float* p) {
  ofstream fs("cavity_cpp_results.txt");

  // u
  fs << "u ";
  for (int i = 0; i < MAT_SIZE; i++) fs << u[i] << " ";
  fs << "\n";
  fs << "v ";
  for (int i = 0; i < MAT_SIZE; i++) fs << v[i] << " ";
  fs << "\n";
  fs << "p ";
  for (int i = 0; i < MAT_SIZE; i++) fs << p[i] << " ";
  fs << "\n";

  fs.close();
}

void build_up_b(float *u, float *v, float *b) {
  for(int i = 1; i < ny - 1; i++){
    for(int j = 1; j < nx - 1; j++){
      b[i*nx+j] = (rho * ( 1.0 / dt *
            ((u[i*nx+j+1] - u[i*nx+j-1]) /
             (2.0 * dx) + (v[(i+1)*nx+j] - v[(i-1)*nx+j]) / (2.0 * dy)) -
            ((u[i*nx+j+1] - u[i*nx+j-1]) / (2.0 * dx)) * ((u[i*nx+j+1] - u[i*nx+j-1]) / (2.0 * dx)) -
            2.0 * ((u[(i+1)*nx+j] - u[(i-1)*nx+j]) / (2.0 * dy) *
              (v[i*nx+j+1] - v[i*nx+j-1]) / (2.0 * dx)) -
            ((v[(i+1)*nx+j] - v[(i-1)*nx+j]) / (2.0 * dy)) * ((v[(i+1)*nx+j] - v[(i-1)*nx+j]) / (2.0 * dy))));
    }
  }
}

void pressure_poisson(float *p, float *b){
  float pn[MAT_SIZE];

  for (int q = 0; q < nit; q++) {
    for (int i = 1; i < ny-1; i++) {
      for (int j = 1; j < nx-1; j++) {
        pn[i*nx+j] = p[i*nx+j];
      }
    }

    for (int i = 1; i < ny-1; i++) {
      for (int j = 1; j < nx-1; j++) {
        p[i*nx+j] = (((pn[i*nx+j+1] + pn[i*nx+j-1]) * pow(dy, 2) +
              (pn[(i+1)*nx+j] + pn[(i-1)*nx+j]) * pow(dx, 2)) /
            (2 * (pow(dx, 2) + pow(dy, 2))) -
            pow(dx, 2) * pow(dy, 2) / (2 * (pow(dx, 2) + pow(dy, 2))) *
            b[i*nx+j]);
      }
    }

    for (int i = 0; i < ny; i++) {
      p[i*nx+nx-1] = p[i*nx+nx-2];
      p[i*nx] = p[i*nx+1];
    }

    for (int i = 0; i < nx; i++) {
      p[i] = p[nx+i];
      p[nx*(ny-1)+i] = 0.0;
    }
  }
}

void cavity_flow(float *u, float *v, float *p) {
  float un[MAT_SIZE], vn[MAT_SIZE], b[MAT_SIZE];

  for (int i = 0; i < MAT_SIZE; i++) {
    un[i] = u[i];
    vn[i] = v[i];
    b[i] = 0.0;
  }

  for (int n = 0; n < nt; n++) {
    for (int i = 0; i < MAT_SIZE; i++) {
      un[i] = u[i];
      vn[i] = v[i];
    }

    build_up_b(u, v, b);
    pressure_poisson(p, b);

    for (int i = 1; i < ny-1; i++) {
      for (int j = 1; j < nx - 1; j++) {
        u[i*nx+j] = (un[i*nx+j] -
          un[i*nx+j] * dt / dx *
          (un[i*nx+j] - un[i*nx+j-1]) -
          vn[i*nx+j] * dt / dy *
          (un[i*nx+j] - un[(i-1)*nx+j]) -
          dt/ (2.0 * rho * dx) * (p[i*nx+j+1] - p[i*nx+j-1]) +
          nu * (dt / pow(dx, 2.0) *
            (un[i*nx+j+1] - 2.0 * un[i*nx+j] + un[i*nx+j-1]) +
            dt / pow(dy, 2.0) *
            (un[(i+1)*nx+j] - 2.0 * un[i*nx+j] + un[(i-1)*nx+j])));

        v[i*nx+j] = (vn[i*nx+j] -
            vn[i*nx+j] * dt / dx *
            (vn[i*nx+j] - vn[i*nx+j-1]) -
            vn[i*nx+j] * dt / dy *
            (vn[i*nx+j] - vn[(i-1)*nx+j]) -
            dt / (2 * rho * dy) * (p[(i+1)*nx+j] - p[(i-1)*nx+j]) +
            nu * (dt / pow(dx, 2) *
              (vn[i*nx+j+1] - 2 * vn[i*nx+j] + vn[i*nx+j-1]) +
              dt / pow(dy, 2) *
              (vn[(i+1)*nx+j] - 2.0 * vn[i*nx+j] + vn[(i-1)*nx+j])));
      }
    }

    for (int i = 0; i < nx; i++) {
      u[i] = 0.0;
      u[(ny-1)*nx+i] = 1.0;
      v[i] = 0.0;
      v[(ny-1)*nx+i] = 0.0;
    }

    for (int i = 0; i < ny; i++) {
      u[i*nx] = 0.0;
      u[i*nx+nx-1] = 0.0;
      v[i*nx] = 0.0;
      v[i*nx+nx-1] = 0.0;
    }

  }
}

int main() {
  float u[MAT_SIZE], v[MAT_SIZE], p[MAT_SIZE];

  for (int i = 0; i < MAT_SIZE; i++) {
    u[i] = 0.0;
    v[i] = 0.0;
    p[i] = 0.0;
  }

  cavity_flow(u, v, p);

  writeFile(u, v, p);

  return 0;
}
