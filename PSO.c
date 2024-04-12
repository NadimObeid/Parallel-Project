#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define N_PARTICLES 1000
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

// function we need to find the global minimum for.
double f(double x, double y) {
    return pow(x - 3.14, 2) + pow(y - 2.72, 2) + sin(3 * x + 1.41) + sin(4 * y - 1.73);
}

void update(particle particles[N_PARTICLES], double gbest[2], double* gbest_obj) {
    double r1 = (double)rand() / RAND_MAX;
    double r2 = (double)rand() / RAND_MAX;

    for (int i = 0; i < N_PARTICLES; i++) {
        particle *curr_particle = &particles[i];
        curr_particle->x_velo = W * curr_particle->x_velo + C1 * r1 * (curr_particle->x_best - curr_particle->x_pos) + C2 * r2 * (gbest[0] - curr_particle->x_pos);
        curr_particle->y_velo = W * curr_particle->y_velo + C1 * r1 * (curr_particle->y_best - curr_particle->y_pos) + C2 * r2 * (gbest[1] - curr_particle->y_pos);

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
    for (int i = 1; i < N_PARTICLES; i++) {
        if (particles[i].best < particles[min_index].best) {
            min_index = i;
        }
    }

    if (particles[min_index].best < *gbest_obj) {
        *gbest_obj = particles[min_index].best;
        gbest[0] = particles[min_index].x_pos;
        gbest[1] = particles[min_index].y_pos;
    }
}

int main() {
    srand(time(NULL));
    // Initialize particles
    particle particles[N_PARTICLES];
    double gbest[2];
    double gbest_obj;
    double total_time;
    for(int k = 0; k< 10; k++){
    // initialize particles
        clock_t start = clock();
        gbest[0] = 0;
        gbest[1] = 0;
        gbest_obj = INFINITY;
        for (int i = 0; i < N_PARTICLES; i++) {
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
            update(particles, gbest, &gbest_obj);
        }
        clock_t end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }
    printf("PSO found best solution at f(%lf,%lf)=%lf\n", gbest[0], gbest[1], gbest_obj);
    printf("Average time is: %f s\n", total_time/10);

    return 0;
}
