#include "cuda_runtime.h"

#include <iostream>
#include <cstdlib>
#include "model.cu"
#include "framework.h"

using namespace std;

class MyModel : public Model{
public:
    __host__ bool validate_cpu(float *point){
        return point[0] > 0 && point[1] > 0;
    }

    __device__ bool validate_gpu(float *point){
        return point[0] > 0 && point[1] > 0;
    }

    bool toBool(){ return true; }
};

int main(){
    int result;
    MyModel model;
    ParallelFrameworkParameters parameters;
    Limit limits[2];

    // Create a model object
    model = MyModel();

    // Create the parameters struct
    parameters.D = 2;
    parameters.batchSize = 1;
    parameters.computeBatchSize = 1;

    // Create the limits for each dimension
    limits[0] = Limit { -10, 10, 1 };
    limits[1] = Limit { -10, 10, 2 };

    // Declare the framework object
    ParallelFramework framework;
     
    // Initialize the framework object
    result = framework.init(limits, parameters, model);
    if (result != 0) {
        cout << "Error initializing framework: " << result << endl;
    }

    // Start the computation
    result = framework.run();
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
    }

    return 0;
}
