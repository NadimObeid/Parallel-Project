#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N_PARTICLES 1000
#define W 0.8
#define C1 0.1
#define C2 0.1
#define N_ITERATIONS 500
#define N 4

typedef struct {
    double x_pos;
    double y_pos;
    double x_velo;
    double y_velo;
    double x_best;
    double y_best;
    double best;
} particle;

double f(double x, double y) {
    return pow(x - 3.14, 2) + pow(y - 2.72, 2) + sin(3 * x + 1.41) + sin(4 * y - 1.73);
}

int main() {
    srand((unsigned int)time(NULL));
    particle particles[N_PARTICLES];
    double gbest[2];
    double gbest_obj;
    double total_time = 0.0;

    for (int k = 0; k < 10; k++) {
        clock_t start = clock();
        gbest[0] = 0;
        gbest[1] = 0;
        gbest_obj = INFINITY;

        #pragma omp parallel num_threads(N)
        {
            double local_gbest_obj = INFINITY;
            double local_gbest[2] = {0, 0};
            double r1, r2;
            particle local_particles[N_PARTICLES / N];

            #pragma omp for
            for (int i = 0; i < N_PARTICLES; i++) {
                particle new_particle;
                new_particle.x_pos = (double)rand() / RAND_MAX * 5;
                new_particle.y_pos = (double)rand() / RAND_MAX * 5;
                new_particle.x_velo = (double)rand() / RAND_MAX * 0.1;
                new_particle.y_velo = (double)rand() / RAND_MAX * 0.1;
                new_particle.x_best = new_particle.x_pos;
                new_particle.y_best = new_particle.y_pos;
                new_particle.best = f(new_particle.x_pos, new_particle.y_pos);
                local_particles[i % (N_PARTICLES / N)] = new_particle;
            }

            for (int i = 0; i < N_ITERATIONS; i++) {
                #pragma omp single
                {
                    r1 = (double)rand() / RAND_MAX;
                    r2 = (double)rand() / RAND_MAX;
                }

                #pragma omp for
                for (int j = 0; j < N_PARTICLES / N; j++) {
                    particle *local_particle = &local_particles[j];
                    local_particle->x_velo = W * local_particle->x_velo + C1 * r1 * (local_particle->x_best - local_particle->x_pos) + C2 * r2 * (gbest[0] - local_particle->x_pos);
                    local_particle->y_velo = W * local_particle->y_velo + C1 * r1 * (local_particle->y_best - local_particle->y_pos) + C2 * r2 * (gbest[1] - local_particle->y_pos);

                    local_particle->x_pos += local_particle->x_velo;
                    local_particle->y_pos += local_particle->y_velo;

                    double obj = f(local_particle->x_pos, local_particle->y_pos);
                    if (local_particle->best > obj) {
                        local_particle->x_best = local_particle->x_pos;
                        local_particle->y_best = local_particle->y_pos;
                        local_particle->best = obj;
                    }

                    if (local_particle->best < local_gbest_obj) {
                        local_gbest_obj = local_particle->best;
                        local_gbest[0] = local_particle->x_pos;
                        local_gbest[1] = local_particle->y_pos;
                    }
                }

                #pragma omp critical
                {
                    if (local_gbest_obj < gbest_obj) {
                        gbest_obj = local_gbest_obj;
                        gbest[0] = local_gbest[0];
                        gbest[1] = local_gbest[1];
                    }
                }
            }
        }

        clock_t end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    printf("PSO found best solution at f(%lf,%lf)=%lf\n", gbest[0], gbest[1], gbest_obj);
    printf("Average time is: %f s\n", total_time / 10);

    return 0;
}