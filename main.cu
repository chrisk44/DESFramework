#include "cuda_runtime.h"

#include <iostream>
#include <cstdlib>

#include "framework.h"

using namespace std;

class MyModel : public Model{
public:
    __host__ bool validate_cpu(float *point){
        return point[0] >= 0 && point[1] >= 0;
    }

    __device__ bool validate_gpu(float *point){
        return point[0] >= 0 && point[1] >= 0;
    }

    bool toBool(){ return true; }
};

int main(){
    int result = 0;
    ParallelFrameworkParameters parameters;
    Limit limits[2];

    // Create the parameters struct
    parameters.D = 2;
    parameters.batchSize = 1;
    parameters.computeBatchSize = 1;

    // Create the limits for each dimension (lower is inclusive, upper is exclusive)
    limits[0] = Limit { -10, 10, 10 };
    limits[1] = Limit { -10, 10, 20 };
     
    // Initialize the framework object
    ParallelFramework framework = ParallelFramework(limits, parameters);
    if (result != 0) {
        cout << "Error initializing framework: " << result << endl;
    }

    // Start the computation
    result = framework.run<MyModel>();
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
    }

    /*
    float point[2];
    float step[2] = {
        abs(limits[0].lowerLimit - limits[0].upperLimit) / limits[0].N,
        abs(limits[1].lowerLimit - limits[1].upperLimit) / limits[1].N
    };
    for (int i = 0; i < limits[0].N; i++) {
        for (int j = 0; j < limits[1].N; j++) {
            point[0] = limits[0].lowerLimit + i * step[0];
            point[1] = limits[1].lowerLimit + j * step[1];
            
            bool result = framework.getResults[framework.getIndexForPoint(point)];
            bool expected = model.validate_cpu(point);

            if (result != expected)
                cout << "ERROR: Point (" << point[0] << "," << point[1] << ") returned " << result << ", expected " << expected << endl;
        }
    }
    */
    
    return 0;
}
