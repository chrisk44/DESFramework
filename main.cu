#include <iostream>
#include <cstdlib>

#include "framework.h"

#define RESULTS_THRESHOLD 1e-13

using namespace std;

class MyModel : public Model{
public:
    __host__ RESULT_TYPE validate_cpu(DATA_TYPE* point, void* dataPtr){
        DATA_TYPE x = point[0];
        DATA_TYPE y = point[1];
        // return sin(x) * sin(y) + pow(x, 2) + pow(y, 2) + x + y + *((int*)dataPtr);
        return x + y;
    }

    __device__ RESULT_TYPE validate_gpu(DATA_TYPE* point, void* dataPtr){
        DATA_TYPE x = point[0];
        DATA_TYPE y = point[1];
        // return sin(x) * sin(y) + pow(x, 2) + pow(y, 2) + x + y + *((int*)dataPtr);
        return x + y;
    }

    bool toBool(RESULT_TYPE result){
        return result >= 8;
    }
};

int main(int argc, char** argv){
    // Initialize framework
    int result = 0;
    ParallelFrameworkParameters parameters;
    Limit limits[2];

    int extraData = 1234;

    // Create the parameters struct
    parameters.D = 2;
    parameters.processingType = PROCESSING_TYPE_BOTH;
    parameters.dataPtr = &extraData;
    parameters.dataSize = sizeof(extraData);

    // Benchmark configuration
    // parameters.resultSaveType = SAVE_TYPE_ALL;
    // parameters.batchSize = 500000000;
    // parameters.benchmark = true;
    // parameters.threadBalancing = true;
    // parameters.slaveBalancing = true;

    // Create the limits for each dimension (lower is inclusive, upper is exclusive)
    // limits[0] = Limit { 0, 10, 50000000 };
    // limits[1] = Limit { -1e05, 1e05, 3000 };

    // Results test configuration
    parameters.resultSaveType = SAVE_TYPE_ALL;
    parameters.batchSize = 20000000;
    parameters.benchmark = false;
    // Create the limits for each dimension (lower is inclusive, upper is exclusive)
    limits[0] = Limit { 0, 10, 50000 };
    limits[1] = Limit { -1e05, 1e05, 3000 };

    // Manual test configuration
    // parameters.resultSaveType = SAVE_TYPE_LIST;
    // parameters.batchSize = 200000;
    // parameters.benchmark = false;
    // // Create the limits for each dimension (lowe is inclusive, upper is exclusive)
    // limits[0] = Limit { 0, 1, 10 };
    // limits[1] = Limit { 0, 10, 10 };

    // Initialize the framework object
    ParallelFramework framework = ParallelFramework(limits, parameters);
    if (! framework.isValid()) {
        cout << "Error initializing framework: " << result << endl;
    }

    // Start the computation
    Stopwatch sw;
    sw.reset();
    sw.start();
    result = framework.run<MyModel>();
    sw.stop();
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
    }
    printf("[Main] Time: %f ms\n", sw.getMsec());

    fflush(stdout);
    if (!parameters.benchmark) {
        // Test the outputs

        unsigned long linearIndex;
        DATA_TYPE point[2];
        DATA_TYPE step[2] = {
            abs(limits[0].lowerLimit - limits[0].upperLimit) / limits[0].N,
            abs(limits[1].lowerLimit - limits[1].upperLimit) / limits[1].N
        };
        MyModel model = MyModel();

        if(parameters.resultSaveType == SAVE_TYPE_ALL){

            DATA_TYPE returned, expected, absError, relError;
            DATA_TYPE absErrorSum = 0, relErrorSum = 0;
            DATA_TYPE absMaxError = 0, relMaxError = 0;
            long skippedInf = 0, skippedNan = 0;

            printf("\nVerifying results...\n");

            for (unsigned int i = 0; i < limits[0].N; i++) {
                point[0] = limits[0].lowerLimit + i * step[0];

                for (unsigned int j = 0; j < limits[1].N; j++) {
                    point[1] = limits[1].lowerLimit + j * step[1];

                    linearIndex = framework.getIndexFromPoint(point);

                    returned = framework.getResults()[linearIndex];
                    expected = model.validate_cpu(point, parameters.dataPtr);

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

        }else{

            int length, correctCount = 0;
            bool shouldBeInList, isInList;
            DATA_TYPE* list = framework.getList(&length);

            printf("\nVerifying results...\n");

            for (unsigned int i = 0; i < limits[0].N; i++) {
                point[0] = limits[0].lowerLimit + i * step[0];

                for (unsigned int j = 0; j < limits[1].N; j++) {
                    point[1] = limits[1].lowerLimit + j * step[1];

                    // Check if it SHOULD be in list
                    shouldBeInList = model.toBool(model.validate_cpu(point, parameters.dataPtr));
                    if(shouldBeInList)
                        correctCount++;

                    // Check the list to see if the point is there
                    isInList = false;
                    for(int k=0; k<length; k++){
                        if(list[k*2] == point[0] && list[k*2 + 1] == point[1]){
                            isInList = true;
                            break;
                        }
                    }

                    if(shouldBeInList ^ isInList)
                        printf("(%f, %f): shouldBeInList=%s, isInList=%s\n", point[0], point[1], shouldBeInList ? "true" : "false", isInList ? "true" : "false");
                }
            }

            if(correctCount != length){
                printf("List length is %d but should be \n", length, correctCount);

                printf("Points is list:\n");
                for(int k=0; k<length; k++){
                    printf("%f, %f\n", list[2*k], list[2*k + 1]);
                }
            }

        }
    }

    return 0;
}
