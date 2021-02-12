#include <iostream>
#include <cstdlib>
#include <fstream>
#include <mpi.h>

#include "framework.h"

/*
 * Each one of these contain a __host__ __device__ doValidate[MO][123] function
*/
#include "mogi1.h"
#include "mogi2.h"
#include "okada1.h"
#include "okada2.h"
#include "okada3.h"

__host__   RESULT_TYPE validate_cpuM1(DATA_TYPE* x, void* dataPtr){ return doValidateM1(x, dataPtr); }
__device__ RESULT_TYPE validate_gpuM1(DATA_TYPE* x, void* dataPtr){ return doValidateM1(x, dataPtr); }

__host__   RESULT_TYPE validate_cpuM2(DATA_TYPE* x, void* dataPtr){ return doValidateM2(x, dataPtr); }
__device__ RESULT_TYPE validate_gpuM2(DATA_TYPE* x, void* dataPtr){ return doValidateM2(x, dataPtr); }

__host__   RESULT_TYPE validate_cpuO1(DATA_TYPE* x, void* dataPtr){ return doValidateO1(x, dataPtr); }
__device__ RESULT_TYPE validate_gpuO1(DATA_TYPE* x, void* dataPtr){ return doValidateO1(x, dataPtr); }

__host__   RESULT_TYPE validate_cpuO2(DATA_TYPE* x, void* dataPtr){ return doValidateO2(x, dataPtr); }
__device__ RESULT_TYPE validate_gpuO2(DATA_TYPE* x, void* dataPtr){ return doValidateO2(x, dataPtr); }

// __host__   RESULT_TYPE validate_cpuO3(DATA_TYPE* x, void* dataPtr){ return doValidateO2(x, dataPtr); }
// __device__ RESULT_TYPE validate_gpuO4(DATA_TYPE* x, void* dataPtr){ return doValidateO2(x, dataPtr); }

__host__ bool toBool_cpu(RESULT_TYPE result){ return result != 0; }
__device__ bool toBool_gpu(RESULT_TYPE result){ return result != 0; }

#define RESULTS_THRESHOLD 1e-13

using namespace std;

int main(int argc, char** argv){
    const char* modelNames[4] = {
        "mogi1",
        "mogi2",
        "okada1",
        "okada2"
    };

    // Scale factor, must be >0
    float k = 2;

    char displFilename[1024];
    char gridFilename[1024];
    char outFilename[1024];
    bool isMaster;
    ifstream dispfile, gridfile;
    ofstream outfile;
    string tmp;
    int stations, dims, i, j, m, g, result, rank;
    float *modelDataPtr, *dispPtr;
    float x, y, z, de, dn, dv, se, sn, sv;
    float low, high, step;

    Stopwatch sw;
    int length;
    DATA_TYPE* list;

    bool onlyOne = false;
    int startModel = 0;
    int endModel = 3;
    int startGrid = 1;
    int endGrid = 6;

    // Initialize MPI manually
    printf("Initializing MPI\n");
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[%d] MPI Initialized\n", rank);
    isMaster = rank == 0;

    // For each model...
    for(m=startModel; m<=endModel; m++){
        printf("[%d] Starting model %d/4...\n", rank, m+1);
        // Open displacements file
        sprintf(displFilename, "./data/%s/displ.txt", modelNames[m]);
        dispfile.open(displFilename, ios::in);

        // Count stations
        stations = 0;
        while(getline(dispfile, tmp)) stations++;

        if(stations < 1){
            printf("[%d][%s \\ %d] Got 0 displacements. Exiting.\n", rank, modelNames[m], g);
            exit(2);
        }

        // Reset the file
        dispfile.close();
        dispfile.open(displFilename, ios::in);

        // Create the model's parameters struct (the model's input data)
        modelDataPtr = new float[1 + stations * (9 - (m<2 ? 0 : 1))];
        modelDataPtr[0] = (float) stations;

        // Read each station's displacement data
        dispPtr = &modelDataPtr[1];
        i = 0;
        if(m < 2){
            // Mogi models have x,y,z,...
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
        }else{
            // Okada models have x,y,...
            while(dispfile >> x >> y >> de >> dn >> dv >> se >> sn >> sv){
                dispPtr[0*stations + i] = x;
                dispPtr[1*stations + i] = y;
                dispPtr[2*stations + i] = de;
                dispPtr[3*stations + i] = dn;
                dispPtr[4*stations + i] = dv;
                dispPtr[5*stations + i] = se * k;
                dispPtr[6*stations + i] = sn * k;
                dispPtr[7*stations + i] = sv * k;

                i++;
            }
        }

        dispfile.close();

        // For each grid...
        for(g=startGrid; g<=endGrid; g++){
            printf("[%d] Starting grid %d/6\n", rank, g);
            if(m == 3 && g > 4)
                continue;

            // Open grid file
            sprintf(gridFilename, "./data/%s/grid%d.txt", modelNames[m], g);
            gridfile.open(gridFilename, ios::in);

            // Count dimensions
            dims = 0;
            while(getline(gridfile, tmp)) dims++;

            // Reset the file
            gridfile.close();
            gridfile.open(gridFilename, ios::in);

            // Read each dimension's grid information
            Limit limits[dims];
            unsigned long totalElements = 1;
            i = 0;
            while(gridfile >> low >> high >> step){
                // Create the limit (lower is inclusive, upper is exclusive)
                high += step;
                limits[i] = Limit{ low, high, (unsigned int) ((high-low)/step) };
                totalElements *= limits[i].N;
                i++;
            }

            // Close the file
            gridfile.close();

            // Create the framework's parameters struct
            ParallelFrameworkParameters parameters;
            parameters.D = dims;
            parameters.resultSaveType = SAVE_TYPE_LIST;
            parameters.processingType = PROCESSING_TYPE_GPU;
            parameters.overrideMemoryRestrictions = true;
            parameters.finalizeAfterExecution = false;
            parameters.printProgress = false;
            parameters.benchmark = false;

            parameters.dataPtr = (void*) modelDataPtr;
            parameters.dataSize = (1 + stations*(9 - (m<2 ? 0 : 1))) * sizeof(float);

            parameters.threadBalancing          = true;
            parameters.slaveBalancing           = true;
            parameters.slaveDynamicScheduling   = true;
            parameters.cpuDynamicScheduling     = true;
            parameters.threadBalancingAverage   = true;

            parameters.batchSize                = ULONG_MAX;
            parameters.slaveBatchSize           = totalElements / 64;
            parameters.computeBatchSize         = 20;
            parameters.cpuComputeBatchSize      = 1e+04;

            parameters.blockSize                = 1024;
            parameters.gpuStreams               = 8;

            parameters.slowStartLimit           = 6;
            parameters.slowStartBase            = 5e+05;
            parameters.minMsForRatioAdjustment  = 10;

            float totalTime = 0;        //msec
            int numOfRuns = 0;
            int numOfResults = -2;
            // Run at least 10 seconds, and stop after 10 runs or 2 minutes
            while(true){
                // Initialize the framework object
                ParallelFramework* framework = new ParallelFramework(false);
                framework->init(limits, parameters);
                if (! framework->isValid()) {
                    printf("[%s \\ %d] Error initializing framework\n", modelNames[m], g);
                    exit(-1);
                }

                // Start the computation
                sw.start();
                switch(m){
                    case 0: framework->run<validate_cpuM1, validate_gpuM1, toBool_cpu, toBool_gpu>(); break;
                    case 1: framework->run<validate_cpuM2, validate_gpuM2, toBool_cpu, toBool_gpu>(); break;
                    case 2: framework->run<validate_cpuO1, validate_gpuO1, toBool_cpu, toBool_gpu>(); break;
                    case 3: framework->run<validate_cpuO2, validate_gpuO2, toBool_cpu, toBool_gpu>(); break;
                }
                sw.stop();
                if (result != 0) {
                    printf("[%s \\ %d] Error running the computation: %d\n", modelNames[m], g, result);
                    exit(-1);
                }

                if(isMaster){
                    framework->getList(&length);
                    if(length != numOfResults && numOfResults != -2){
                        printf("[%s \\ %d] Number of results from run %d don't match: %d -> %d.\n",
                                        modelNames[m], g, numOfRuns, numOfResults, length);
                    }
                    numOfResults = length;
                    // printf("[%s \\ %d] Run %d: %f ms, %d results\n", modelNames[m], g, numOfRuns, length, sw.getMsec());
                }

                totalTime += sw.getMsec();
                numOfRuns++;

                int next;
                if(isMaster){
                    if(onlyOne || (totalTime > 10 * 1000 && (numOfRuns >= 10 || totalTime >= 1 * 60 * 1000))){
                        printf("[%s \\ %d] Time: %f ms in %d runs\n",
                                    modelNames[m], g, totalTime/numOfRuns, numOfRuns);

                        sprintf(outFilename, "results_%s_%d.txt", modelNames[m], g);
                        // Open file to write results
                        outfile.open(outFilename, ios::out | ios::trunc);

                        list = framework->getList(&length);
                        printf("[%s \\ %d] Results: %d\n", modelNames[m], g, length);
                        for(i=0; i<min(length, 5); i++){
                            printf("[%s \\ %d] (", modelNames[m], g);
                            for(j=0; j<parameters.D-1; j++)
                                printf("%lf ", list[i*parameters.D + j]);

                            printf("%lf)\n", list[i*parameters.D + j]);
                        }
                        if(length > 5)
                            printf("[%s \\ %d] ...%d more results\n", modelNames[m], g, length-5);

                        printf("\n");
                        if(g==6)
                            printf("\n");

                        for(i=0; i<length; i++){
                            for(j=0; j<parameters.D-1; j++)
                                outfile << list[i*parameters.D + j] << " ";

                            outfile << list[i*parameters.D + j] << endl;
                        }

                        outfile.close();
                        next = 1;
                    }else{
                        next = 0;
                    }
                }

                delete framework;
                MPI_Bcast(&next, 1, MPI_INT, 0, MPI_COMM_WORLD);

                if(next)
                    break;
            }

            if(onlyOne)
                break;
        }

        delete [] modelDataPtr;

        if(onlyOne)
            break;
    }

    MPI_Finalize();

    return 0;
}
