#include "cuda_runtime.h"

#include <iostream>
#include <cstdlib>
#include <mpi.h>

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

int main(int argc, char** argv){
    /*
    int result = 0;
    ParallelFrameworkParameters parameters;
    Limit limits[2];

    // Create the parameters struct
    parameters.D = 2;
    parameters.batchSize = 199;
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

    long linearIndex;
    long indices[2];
    float point[2];
    float step[2] = {
        abs(limits[0].lowerLimit - limits[0].upperLimit) / limits[0].N,
        abs(limits[1].lowerLimit - limits[1].upperLimit) / limits[1].N
    };
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
            }
        }
    } */

    int i, rank, size, namelen;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status stat;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &namelen);

    if (rank == 0) {
        printf("Hello world: rank %d of %d running on %s\n", rank, size, name);

        for (i = 1; i < size; i++) {
            MPI_Recv(&rank, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(&size, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(&namelen, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &stat);
            MPI_Recv(name, namelen + 1, MPI_CHAR, i, 1, MPI_COMM_WORLD, &stat);
            printf("Hello world: rank %d of %d running on %s\n", rank, size, name);
        }
    } else {
        MPI_Send(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&namelen, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(name, namelen + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    
    return 0;
}
