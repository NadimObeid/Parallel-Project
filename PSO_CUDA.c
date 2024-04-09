#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define N_PARTICLES 20
#define W 0.8
#define C1 0.1
#define C2 0.1
#define N_ITERATIONS 50

typedef struct{
    float x_pos;
    float y_pos;
    float x_velo;
    float y_velo;
    float x_best;
    float y_best;
    float best;

}particle;

double f(double x, double y) {
    return pow(x - 3.14, 2) + pow(y - 2.72, 2) + sin(3 * x + 1.41) + sin(4 * y - 1.73);
}


// CUDA kernel to update particle positions and velocities
_global_ void update(double *X, double *V, double *pbest, double *pbest_obj, double *gbest, double *gbest_obj) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N_PARTICLES) {
        double r1 = (double)rand() / RAND_MAX;
        double r2 = (double)rand() / RAND_MAX;

        V[i] = W * V[i] + C1 * r1 * (pbest[i] - X[i]) + C2 * r2 * (gbest[i / 2] - X[i]);

        X[i] += V[i];

        double obj = f(X[i],X[i+N_PARTICLES]);
        if (pbest_obj[i] > obj) {
            pbest[i] = X[i];
            pbest_obj[i] = obj;
        }

        __syncthreads();

        if (pbest_obj[i] < *gbest_obj) {
            atomicMin((unsigned long long int *)gbest_obj, (unsigned long long int)pbest_obj[i]);
            gbest[i / 2] = pbest[i];
        }
    }
}

int main() {
    srand(time(NULL));

    // Initialize host variables
    double X_host[2 * N_PARTICLES];
    double V_host[2 * N_PARTICLES];
    double pbest_host[2 * N_PARTICLES];
    double pbest_obj_host[N_PARTICLES];
    double gbest_host[2];
    double gbest_obj_host = INFINITY;

    // Initialize device variables
    double *X_dev, *V_dev, *pbest_dev, *pbest_obj_dev, *gbest_dev, *gbest_obj_dev;
    cudaMalloc((void **)&X_dev, 2 * N_PARTICLES * sizeof(double));
    cudaMalloc((void **)&V_dev, 2 * N_PARTICLES * sizeof(double));
    cudaMalloc((void **)&pbest_dev, 2 * N_PARTICLES * sizeof(double));
    cudaMalloc((void **)&pbest_obj_dev, N_PARTICLES * sizeof(double));
    cudaMalloc((void **)&gbest_dev, 2 * sizeof(double));
    cudaMalloc((void **)&gbest_obj_dev, sizeof(double));

    // Initialize particles
    for (int i = 0; i < N_PARTICLES; i++) {
        X_host[i] = (double)rand() / RAND_MAX * 5;
        X_host[i + N_PARTICLES] = (double)rand() / RAND_MAX * 5;
        V_host[i] = (double)rand() / RAND_MAX * 0.1;
        V_host[i + N_PARTICLES] = (double)rand() / RAND_MAX * 0.1;
        pbest_host[i] = X_host[i];
        pbest_host[i + N_PARTICLES] = X_host[i + N_PARTICLES];
        pbest_obj_host[i] = pow(X_host[i] - 3.14, 2) + pow(X_host[i + N_PARTICLES] - 2.72, 2) + sin(3 * X_host[i] + 1.41) + sin(4 * X_host[i + N_PARTICLES] - 1.73);
    }

    // Copy data from host to device
    cudaMemcpy(X_dev, X_host, 2 * N_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(V_dev, V_host, 2 * N_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pbest_dev, pbest_host, 2 * N_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pbest_obj_dev, pbest_obj_host, N_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gbest_dev, gbest_host, 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gbest_obj_dev, &gbest_obj_host, sizeof(double), cudaMemcpyHostToDevice);

    // PSO iterations
    for (int i = 0; i < N_ITERATIONS; i++) {
        update<<<(N_PARTICLES + 255) / 256, 256>>>(X_dev, V_dev, pbest_dev, pbest_obj_dev, gbest_dev, gbest_obj_dev);
    }

    // Copy data from device to host
    cudaMemcpy(&gbest_host, gbest_dev, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gbest_obj_host, gbest_obj_dev, sizeof(double), cudaMemcpyDeviceToHost);

    printf("PSO found best solution at f(%lf,%lf)=%lf\n", gbest_host[0], gbest_host[1], gbest_obj_host);

    // Free device memory
    cudaFree(X_dev);
    cudaFree(V_dev);
    cudaFree(pbest_dev);
    cudaFree(pbest_obj_dev);
    cudaFree(gbest_dev);
    cudaFree(gbest_obj_dev);

    return 0;
}