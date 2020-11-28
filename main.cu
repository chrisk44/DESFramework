#include <iostream>
#include <cstdlib>
#include <fstream>

#include "framework.h"

/*
 * Each one of these contain a __host__ __device__ doValidate function and have defined a MODEL_D
 * Obviously include only one.
*/
// #include "mogi1.h"
#include "mogi2.h"
// #include "okada1.h"
// #include "okada2.h"
// #include "okada3.h"
// #include "test.h"

__host__ RESULT_TYPE validate_cpu(DATA_TYPE* x, void* dataPtr){
    return doValidate(x, dataPtr);
}

__device__ RESULT_TYPE validate_gpu(DATA_TYPE* x, void* dataPtr){
    return doValidate(x, dataPtr);
}

__host__ bool toBool_cpu(RESULT_TYPE result){
    return result != 0;
}

__device__ bool toBool_gpu(RESULT_TYPE result){
    return result != 0;
}

#define RESULTS_THRESHOLD 1e-13

using namespace std;

int main(int argc, char** argv){
    ifstream dispfile, gridfile;
    ofstream outfile;
    string tmp;
    int stations, dims, i, j, result;
    float *modelDataPtr, *dispPtr;
    float x, y, z, de, dn, dv, se, sn, sv, k;
    float low, high, step;

    ParallelFrameworkParameters parameters;
    Limit limits[MODEL_D];

    Stopwatch sw;
    int length;
    DATA_TYPE* list;

    if(argc != 4){
        printf("This: ");
        for(i=0; i<argc; i++)
            printf("%s ", argv[i]);

        printf("is unacceptable. You wrote this thing.\n");

        printf("[E] Usage: %s <displacements file> <grid file> <k >= 0>\n", argv[0]);
        exit(1);
    }

    k = atof(argv[3]);
    if(k <= 0){
        printf("[E] Scale factor k must be > 0. Exiting.\n");
        exit(1);
    }

    printf("Reading displacements from %s\n", argv[1]);
    printf("Reading grid from %s\n", argv[2]);
    printf("Scale factor = %f\n", k);

    // Open files
    dispfile.open(argv[1], ios::in);
    gridfile.open(argv[2], ios::in);

    stations = 0;
    dims = 0;

    // Count stations and dimensions
    while(getline(dispfile, tmp)) stations++;
    while(getline(gridfile, tmp)) dims++;

    printf("Got %d stations\n", stations);
    printf("Got %d dimensions\n", dims);

    if(stations < 1){
        printf("Got 0 displacements. Exiting.\n");
        exit(2);
    }
    if(dims != MODEL_D){
        printf("Got %d dimensions, expected %d. Exiting.\n", dims, MODEL_D);
        exit(2);
    }

    // Reset the files
    dispfile.close();
    gridfile.close();
    dispfile.open(argv[1], ios::in);
    gridfile.open(argv[2], ios::in);

    // Create the model's parameters struct (the model's input data)
    modelDataPtr = new float[1 + stations*9];
    modelDataPtr[0] = (float) stations;

    // Read each station's displacement data
    dispPtr = &modelDataPtr[1];
    i = 0;
    while(dispfile >> x >> y >> z >> de >> dn >> dv >> se >> sn >> sv){
        dispPtr[0*stations + i] = x;
        dispPtr[1*stations + i] = y;
        dispPtr[2*stations + i] = z;
        dispPtr[3*stations + i] = de;
        dispPtr[4*stations + i] = dn;
        dispPtr[5*stations + i] = dv;
        dispPtr[6*stations + i] = se * k;
        dispPtr[7*stations + i] = sn * k;
        dispPtr[8*stations + i] = sv * k;

        i++;
    }

    // Read each dimension's grid information
    i = 0;
    while(gridfile >> low >> high >> step){
        // Create the limit (lower is inclusive, upper is exclusive)
        high += step;
        limits[i] = Limit{ low, high, (unsigned int) ((high-low)/step) };
        i++;
    }

    dispfile.close();
    gridfile.close();

    // Create the framework's parameters struct
    parameters.D = MODEL_D;
    parameters.resultSaveType = SAVE_TYPE_LIST;
    parameters.processingType = PROCESSING_TYPE_GPU;
    parameters.dataPtr = (void*) modelDataPtr;
    parameters.dataSize = (1 + stations*9) * sizeof(float);
    parameters.computeBatchSize = 200;
    parameters.blockSize = 1024;
    parameters.gpuStreams = 8;
    parameters.overrideMemoryRestrictions = true;

    parameters.benchmark = false;
    parameters.threadBalancing = true;
    parameters.slaveBalancing = true;
    parameters.batchSize = ULONG_MAX;
    parameters.slowStartLimit = 5;

    // Initialize the framework object
    ParallelFramework framework = ParallelFramework();
    framework.init(limits, parameters);
    if (! framework.isValid()) {
        cout << "Error initializing framework: " << endl;
        exit(-1);
    }

    // Start the computation
    sw.start();
    result = framework.run<validate_cpu, validate_gpu, toBool_cpu, toBool_gpu>();
    sw.stop();
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
        exit(-1);
    }
    if(framework.getRank() != 0)
        exit(0);

    printf("Time: %f ms\n", sw.getMsec());

    // Open file to write results
    outfile.open("results.txt", ios::out | ios::trunc);

    list = framework.getList(&length);
    printf("Results: %d\n", length);
    for(i=0; i<min(length, 5); i++){
        printf("(");
        for(j=0; j<parameters.D-1; j++)
            printf("%lf ", list[i*parameters.D + j]);

        printf("%lf)\n", list[i*parameters.D + j]);
    }
    if(length > 5)
        printf("...%d more results\n", length-5);

    for(i=0; i<length; i++){
        for(j=0; j<parameters.D-1; j++)
            outfile << list[i*parameters.D + j] << " ";

        outfile << list[i*parameters.D + j] << endl;
    }

    outfile.close();
    delete [] modelDataPtr;

    return 0;
}
