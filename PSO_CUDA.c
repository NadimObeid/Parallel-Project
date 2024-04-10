#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define N_PARTICLES 1024
#define W 0.8
#define C1 0.1
#define C2 0.1
#define N_ITERATIONS 500

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
__global__ void update(particle *particles, double *gbest, double *gbest_obj) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N_PARTICLES) {
        for(int k = 0; k<N_ITERATIONS; k++){
            double r1 = (double)rand() / RAND_MAX;
            double r2 = (double)rand() / RAND_MAX;

            particles[i].x_velo = W * particles[i].x_velo + C1 * r1 * (particles[i].x_best - particles[i].x_pos) + C2 * r2 * (gbest[0] - particles[i].x_pos);
            particles[i].y_velo = W * particles[i].y_velo + C1 * r1 * (particles[i].y_best - particles[i].y_pos) + C2 * r2 * (gbest[1] - particles[i].y_pos);

            particles[i].x_pos += particles[i].x_velo;
            particles[i].y_pos += particles[i].y_velo;

            double obj = f(particles[i].x_best,particles[i].y_best);
            if (particles[i].best > obj) {
                particles[i].x_best = particles[i].x_pos;
                particles[i].y_best = particles[i].y_pos;
                particles[i].best = obj;
            }

            if (particles[i].best < *gbest_obj) {
                atomicMin((unsigned long long int *)gbest_obj, (unsigned long long int)particles[i].best);
                gbest[0] = particles[i].x_best;
                gbest[1] = particles[i].y_best;
            }
            __syncthreads();
        }
    }
}

int main() {
    srand(time(NULL));

    // Initialize host variables
    particle particles_host[N_PARTICLES];
    double gbest_host[2];
    double gbest_obj_host = INFINITY;

    // Initialize device variables
    particle *particles_dev;
    double *gbest_dev, *gbest_obj_dev;
    cudaMalloc((void **)&particles_dev, N_PARTICLES * sizeof(particle));
    cudaMalloc((void **)&gbest_dev, 2 * sizeof(double));
    cudaMalloc((void **)&gbest_obj_dev, sizeof(double));

    // Initialize particles
    for (int i = 0; i < N_PARTICLES; i++) {
        particles_host[i].x_pos = (double)rand() / RAND_MAX * 5;
        particles_host[i].y_pos = (double)rand() / RAND_MAX * 5;
        particles_host[i].x_velo = (double)rand() / RAND_MAX * 0.1;
        particles_host[i].y_velo = (double)rand() / RAND_MAX * 0.1;
        particles_host[i].x_best = particles_host[i].x_pos;
        particles_host[i].y_best = particles_host[i].y_pos;
        particles_host[i].best= f(particles_host[i].x_best,particles[i].y_best);
    }

    // Copy data from host to device
    cudaMemcpy(particles_dev, particles_host, N_PARTICLES * sizeof(particle), cudaMemcpyHostToDevice);
    cudaMemcpy(gbest_dev, gbest_host, 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gbest_obj_dev, &gbest_obj_host, sizeof(double), cudaMemcpyHostToDevice);

    // PSO iterations
    
    update<<<(N_PARTICLES + 255) / 256, 256>>>(particles_dev, gbest_dev, gbest_obj_dev);

    // Copy data from device to host
    cudaMemcpy(&gbest_host, gbest_dev, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gbest_obj_host, gbest_obj_dev, sizeof(double), cudaMemcpyDeviceToHost);

    printf("PSO found best solution at f(%f,%f)=%f\n", gbest_host[0], gbest_host[1], gbest_obj_host);

    // Free device memory
    cudaFree(particles_dev);
    cudaFree(gbest_dev);
    cudaFree(gbest_obj_dev);

    return 0;
}