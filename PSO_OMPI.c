#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N_PARTICLES 1000
#define W 0.8
#define C1 0.1
#define C2 0.1
#define N_ITERATIONS 500

int size;
typedef struct{
    float x_pos;
    float y_pos;
    float x_velo;
    float y_velo;
    float x_best;
    float y_best;
    float best;

}particle;

// function we need to find the global minimum for.
double f(double x, double y) {
    return pow(x - 3.14, 2) + pow(y - 2.72, 2) + sin(3 * x + 1.41) + sin(4 * y - 1.73);
}

void update(particle particles[N_PARTICLES/size], double gbest[3]) {
    double r1 = (double)rand() / RAND_MAX;
    double r2 = (double)rand() / RAND_MAX;

    for (int i = 0; i < N_PARTICLES/size; i++) {
        particle *curr_particle = &particles[i];
        curr_particle->x_velo = W * curr_particle->x_velo + C1 * r1 * (curr_particle->x_best - curr_particle->x_pos) + C2 * r2 * (gbest[1] - curr_particle->x_pos);
        curr_particle->y_velo = W * curr_particle->y_velo + C1 * r1 * (curr_particle->y_best - curr_particle->y_pos) + C2 * r2 * (gbest[2] - curr_particle->y_pos);

        curr_particle->x_pos += curr_particle->x_velo;
        curr_particle->y_pos += curr_particle->y_velo;

        double obj = f(curr_particle->x_pos, curr_particle->y_pos);
        if (curr_particle->best > obj) {
            curr_particle->x_best = curr_particle->x_pos;
            curr_particle->y_best = curr_particle->y_pos;
            curr_particle->best = obj;
        }
    }

    int min_index = 0;
    for (int i = 1; i < N_PARTICLES/size; i++) {
        if (particles[i].best < particles[min_index].best) {
            min_index = i;
        }
    }

    if (particles[min_index].best < gbest[0]) {
        gbest[0] = particles[min_index].best;
        gbest[1] = particles[min_index].x_best;
        gbest[2] = particles[min_index].y_best;
    }
}

int main() {
    
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank + time(NULL));
    // Initialize particles
    particle particles[N_PARTICLES/size];
    double gbest[3] = {INFINITY, 0, 0};
    double total_time;
    for(int k = 0; k<10; k++){
        clock_t start = clock();
        for (int i = 0; i < N_PARTICLES/size; i++) {
            particles[i].x_pos = (double)rand() / RAND_MAX * 5;
            particles[i].y_pos = (double)rand() / RAND_MAX * 5;
            particles[i].x_velo = (double)rand() / RAND_MAX * 0.1;
            particles[i].y_velo = (double)rand() / RAND_MAX * 0.1;
            particles[i].x_best = particles[i].x_pos;
            particles[i].y_best = particles[i].y_pos;
            particles[i].best = f(particles[i].x_pos, particles[i].y_pos);
        }

        // PSO iterations
        for (int i = 0; i < N_ITERATIONS; i++) {
            update(particles, gbest);
            
        }

        if (rank != 0) {
            MPI_Send(&gbest, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            double process_gbest[3];
            for (int i = 1; i < size; i++) {
                MPI_Recv(&process_gbest, 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (gbest[0] > process_gbest[0]) {
                    gbest[0] = process_gbest[0];
                    gbest[1] = process_gbest[1];
                    gbest[2] = process_gbest[2];
                }
            }
            clock_t end = clock();
            total_time += (double)(end - start) / CLOCKS_PER_SEC;
        }
    }
    MPI_Finalize();
    if(rank == 0){
        printf("PSO found best solution at f(%lf,%lf)=%lf\n", gbest[1], gbest[2], gbest[0]);
        printf("Average time is: %lf s\n", total_time/10);
    }    
    return 0;
}


