#include <iostream>
#include <cstdlib>

#include "framework.h"
#include "mogi_model.h"

#define RESULTS_THRESHOLD 1e-13

using namespace std;

int main(int argc, char** argv){
    // Initialize framework
    int result = 0;

    // Create the model's parameters struct (the model's input data)
    // TBD: If the length of 'displacements' is known at compile time...
    MogiParameters mogiParameters;
    int displacementsLength = 10;
    mogiParameters.stations = 1;
    for(int i=0; i<displacementsLength; i++){
        mogiParameters.displacements[i] = 0;
    }

    // // TBD: If the length of 'displacements is not known at compile time'...
    // int displacementsLength = 10;        // <-- Determine this at runtime
    // int stations = 2;                    // <-- Determine this at runtime
    // float* modelDataPtr = (float*) malloc((displacementsLength+1) * sizeof(float));
    // modelDataPtr[0] = stations;
    // for(int i=0; i<displacementsLength; i++){
    //     modelDataPtr[i+1] = 0;           // <-- Write at index i+1 because the first elements of the array is reserved for 'stations'
    // }

    // Create the framework's parameters struct
    ParallelFrameworkParameters parameters;
    parameters.D = 2;
    parameters.resultSaveType = SAVE_TYPE_LIST;
    parameters.processingType = PROCESSING_TYPE_BOTH;
    parameters.dataPtr = &mogiParameters;
    parameters.dataSize = sizeof(mogiParameters);
    // parameters.dataPtr = (void*) modelDataPtr;
    // parameters.dataSize = (displacementsLength+1) * sizeof(float);
    parameters.threadBalancing = true;
    parameters.slaveBalancing = true;
    parameters.benchmark = false;
    parameters.batchSize = 20000000;

    // Create the limits for each dimension (lower is inclusive, upper is exclusive)
    Limit limits[4];
    limits[0] = Limit { 0, 1, 50000 };
    limits[1] = Limit { 0, 1, 50000 };
    limits[2] = Limit { 0, 1, 50000 };
    limits[3] = Limit { 0, 1, 50000 };

    // Initialize the framework object
    ParallelFramework framework = ParallelFramework(limits, parameters);
    if (! framework.isValid()) {
        cout << "Error initializing framework: " << result << endl;
        exit(-1);
    }

    // Start the computation
    Stopwatch sw;
    sw.reset();
    sw.start();
    result = framework.run<MyModel>();
    sw.stop();
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
        exit(-1);
    }
    printf("[Main] Time: %f ms\n", sw.getMsec());

    int length;
    DATA_TYPE* list = framework.getList(&length);
    printf("Results:\n");
    for(int k=0; k<length; k++){
        printf("(%f, %f, %f, %f)\n", list[4*k], list[4*k + 1], list[4*k + 2], list[4*k + 3]);
    }

    fflush(stdout);
    if (!parameters.benchmark) {
        // Test the outputs

        DATA_TYPE point[2];
        DATA_TYPE step[2] = {
            abs(limits[0].lowerLimit - limits[0].upperLimit) / limits[0].N,
            abs(limits[1].lowerLimit - limits[1].upperLimit) / limits[1].N
        };
        MyModel model = MyModel();

        int correctCount = 0;
        bool shouldBeInList, isInList;
        list = framework.getList(&length);

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

    return 0;
}
