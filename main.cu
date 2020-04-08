#include "cuda_runtime.h"

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <Windows.h>

#include "framework.h"

using namespace std;

class MyModel : public Model{
public:
    __host__ bool validate_cpu(float *point){
        return (point[0] >= 0 || point[0] <= 5) && (point[1] >= 5 || point[1] <= 7.8);
    }

    __device__ bool validate_gpu(float *point){
        return (point[0] >= 0 || point[0] <= 5) && (point[1] >= 5 || point[1] <= 7.8);
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
    parameters.batchSize = 100000;
    parameters.computeBatchSize = 0;
    parameters.cpuOnly = true;
    parameters.dynamicBatchSize = true;

    // Create the limits for each dimension (lower is inclusive, upper is exclusive)
    limits[0] = Limit { -10, 10, 20 };
    limits[1] = Limit { 0, 20, 20};
     
    // Initialize the framework object
    ParallelFramework framework = ParallelFramework(limits, parameters);
    if (result != 0) {
        cout << "Error initializing framework: " << result << endl;
    }

    // Start the computation
    result = framework.run<MyModel>(argv[0]);
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
    }

    Sleep(1000);

    // Test the outputs
    long linearIndex;
    long indices[2];
    float point[2];
    float step[2] = {
        abs(limits[0].lowerLimit - limits[0].upperLimit) / limits[0].N,
        abs(limits[1].lowerLimit - limits[1].upperLimit) / limits[1].N
    };
    int errors = 0;

    cout << endl << "Verifying results..." << endl;
    for (unsigned int i = 0; i < limits[0].N; i++) {
        for (unsigned int j = 0; j < limits[1].N; j++) {
            point[0] = limits[0].lowerLimit + i * step[0];
            point[1] = limits[1].lowerLimit + j * step[1];

            framework.getIndicesFromPoint(point, indices);
            linearIndex = framework.getIndexFromIndices(indices);

            bool result = framework.getResults()[linearIndex];
            bool expected = MyModel().validate_cpu(point);

            if ((!result && expected) || (result && !expected)) {
                cout << "ERROR: Point (" << point[0] << "," << point[1] << ") returned " << result << ", expected " << expected << endl;
                errors++;
            }
        }
    }
    cout << "Done. Errors: " << errors << "/" << limits[0].N * limits[1].N << endl;
    
    return 0;
}
