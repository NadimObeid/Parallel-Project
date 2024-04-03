#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N_PARTICLES 20
#define W 0.8
#define C1 0.1
#define C2 0.1
#define N_ITERATIONS 50

double X[2][N_PARTICLES];
double V[2][N_PARTICLES];
double pbest[2][N_PARTICLES];
double pbest_obj[N_PARTICLES];
double gbest[2];
double gbest_obj = INFINITY;

// function we need to find the global minimum for.
double f(double x, double y) {
    return pow(x - 3.14, 2) + pow(y - 2.72, 2) + sin(3 * x + 1.41) + sin(4 * y - 1.73);
}

void update() {
    double r1 = (double)rand() / RAND_MAX;
    double r2 = (double)rand() / RAND_MAX;

    #pragma omp parallel for
    for (int i = 0; i < N_PARTICLES; i++) {
        V[0][i] = W * V[0][i] + C1 * r1 * (pbest[0][i] - X[0][i]) + C2 * r2 * (gbest[0] - X[0][i]);
        V[1][i] = W * V[1][i] + C1 * r1 * (pbest[1][i] - X[1][i]) + C2 * r2 * (gbest[1] - X[1][i]);

        X[0][i] += V[0][i];
        X[1][i] += V[1][i];

        double obj = f(X[0][i], X[1][i]);
        if (pbest_obj[i] > obj) {
            pbest[0][i] = X[0][i];
            pbest[1][i] = X[1][i];
            pbest_obj[i] = obj;
        }
    }

    int min_index = 0;
    for (int i = 1; i < N_PARTICLES; i++) {
        if (pbest_obj[i] < pbest_obj[min_index]) {
            min_index = i;
        }
    }

    if (pbest_obj[min_index] < gbest_obj) {
        gbest_obj = pbest_obj[min_index];
        gbest[0] = pbest[0][min_index];
        gbest[1] = pbest[1][min_index];
    }
    
}

int main() {
    srand(time(NULL));    

    clock_t start = clock();
    for (int i = 0; i < N_PARTICLES; i++) {
        X[0][i] = (double)rand() / RAND_MAX * 5;
        X[1][i] = (double)rand() / RAND_MAX * 5;
        V[0][i] = (double)rand() / RAND_MAX * 0.1;
        V[1][i] = (double)rand() / RAND_MAX * 0.1;
        pbest[0][i] = X[0][i];
        pbest[1][i] = X[1][i];
        pbest_obj[i] = f(X[0][i], X[1][i]);
    }

    
    // PSO iterations
    for (int i = 0; i < N_ITERATIONS; i++) {
        update();
    }
    clock_t end = clock();
    float total_time = (float)(end-start)/CLOCKS_PER_SEC;

    printf("PSO found best solution at f(%lf,%lf)=%lf\n", gbest[0], gbest[1], gbest_obj);
    printf("Total time is: %f s\n", total_time);

    return 0;
}
