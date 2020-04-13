#include "cuda_runtime.h"

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <sys/time.h>
#include <omp.h>

#include "framework.h"

#define RESULTS_THRESHOLD 1e-15

using namespace std;

class MyModel : public Model{
public:
    __host__ RESULT_TYPE validate_cpu(DATA_TYPE* point){
        DATA_TYPE x = point[0];
        DATA_TYPE y = point[1];
        return sin(x) * sin(y) + pow(x, 2) + pow(y, 2) + x + y;
        //return x + y;
    }

    __device__ RESULT_TYPE validate_gpu(DATA_TYPE* point){
        DATA_TYPE x = point[0];
        DATA_TYPE y = point[1];
        return sin(x) * sin(y) + pow(x, 2) + pow(y, 2) + x + y;
        //return x + y;
    }

    bool toBool(){ return true; }
};

int main(int argc, char** argv){
    // Initialize framework
    int result = 0;
    ParallelFrameworkParameters parameters;
    Limit limits[2];

    // Create the parameters struct
    parameters.D = 2;
    parameters.batchSize = 20000;
    parameters.processingType = TYPE_BOTH;
    parameters.dynamicBatchSize = false;
    parameters.benchmark = false;
    parameters.serverName = "192.168.1.201";        // This would be an input to the program, not relevant to the framework

    // Create the limits for each dimension (lower is inclusive, upper is exclusive)
    limits[0] = Limit { 0, 10, 5000 };
    limits[1] = Limit { -1e05, 1e05, 3000 };

    // Initialize the framework object
    ParallelFramework framework = ParallelFramework(limits, parameters);
    if (! framework.isValid()) {
        cout << "Error initializing framework: " << result << endl;
    }

    // Start the computation
    result = framework.run<MyModel>(argc > 1 ? TYPE_SLAVE : TYPE_MASTER);
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
    }

    fflush(stdout);
    if (!parameters.benchmark) {

        // Test the outputs
        #if DEBUG >= 4
        // Print outputs
        printf("final results: ");
        for (unsigned int i = 0; i < framework.totalElements; i++) {
            printf("%f ", framework.getResults()[i]);
        }
        printf("\n");
        #endif

        unsigned long linearIndex;
        unsigned long indices[2];
        DATA_TYPE point[2];
        DATA_TYPE step[2] = {
            abs(limits[0].lowerLimit - limits[0].upperLimit) / limits[0].N,
            abs(limits[1].lowerLimit - limits[1].upperLimit) / limits[1].N
        };
        DATA_TYPE returned, expected, absError, relError;
        DATA_TYPE absErrorSum = 0, relErrorSum = 0;
        DATA_TYPE absMaxError = 0, relMaxError = 0;
        long skippedInf = 0, skippedNan = 0;

        cout << endl << "Verifying results..." << endl;
        for (unsigned int i = 0; i < limits[0].N; i++) {
            point[0] = limits[0].lowerLimit + i * step[0];

            for (unsigned int j = 0; j < limits[1].N; j++) {
                point[1] = limits[1].lowerLimit + j * step[1];

                framework.getIndicesFromPoint(point, indices);
                linearIndex = framework.getIndexFromIndices(indices);

                returned = framework.getResults()[linearIndex];
                expected = MyModel().validate_cpu(point);

                absError = abs(returned - expected);
                relError = absError == 0 ? 0 : abs(absError / max(abs(expected), abs(returned)));

                if (isinf(absError) || isinf(relError)) {
                    printf("(%f, %f): result=%f, expected=%f, absError=%f, relError=%f\n", point[0], point[1], returned, expected, absError, relError);
                    skippedInf++;
                } else if (isnan(absError) || isnan(relError)) {
                    printf("(%f, %f): result=%f, expected=%f, absError=%f, relError=%f\n", point[0], point[1], returned, expected, absError, relError);
                    skippedNan++;
                } else {
                    if (relError > RESULTS_THRESHOLD) {
                        printf("(%f, %f): result=%f, expected=%f, absError=%f, relError=%f\n", point[0], point[1], returned, expected, absError, relError);
                    }
                    absErrorSum += absError;
                    relErrorSum += relError;
                    absMaxError = max(absMaxError, absError);
                    relMaxError = max(relMaxError, relError);
                }
            }

        }

        printf("Absolute error: Max=%f, Avg=%f\n", absMaxError, absErrorSum / (limits[0].N * limits[1].N));
        printf("Relative error: Max=%f, Avg=%f\n", relMaxError, relErrorSum / (limits[0].N * limits[1].N));
        printf("Skipped elements: Inf=%ld, NaN=%ld\n", skippedInf, skippedNan);
    }

    return 0;
}
