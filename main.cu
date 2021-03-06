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

#define ERR_INVALID_ARG -1

using namespace std;

bool onlyOne = false;
int startModel = 0;
int endModel = 3;
int startGrid = 1;
int endGrid = 6;

ProcessingType processingType = PROCESSING_TYPE_GPU;
bool threadBalancing          = true;
bool slaveBalancing           = true;
bool slaveDynamicScheduling   = true;
bool cpuDynamicScheduling     = true;
bool threadBalancingAverage   = true;

unsigned long batchSize           = UINT_MAX;
unsigned long slaveBatchSize      = 1e+07;
unsigned long computeBatchSize    = 20;
unsigned long cpuComputeBatchSize = 1e+04;

int blockSize  = 1024;
int gpuStreams = 8;

int slowStartLimit          = 6;
unsigned long slowStartBase = 5e+05;
int minMsForRatioAdjustment = 10;

unsigned long getOrDefault(int argc, char** argv, bool* found, int* i, const char* argName, const char* argNameShort, bool requiresMore, unsigned long defaultValue){
    if(*i >= argc)
        return defaultValue;

    if(strcmp(argv[*i], argName) == 0 || strcmp(argv[*i], argNameShort) == 0){
        // If it required more and we have it...
        if(requiresMore && (*i + 1) < argc){
            defaultValue = atoi(argv[*i+1]);
            *i += 2;
        }
        // else if it doesn't require a second argument, so just mark it as 'found'
        else if(!requiresMore){
            defaultValue = 1;
            *i += 1;
        }
        // else if it requires more and we don't have it
        else{
            fprintf(stderr, "[E] %s requires an additional argument\n", argName);
            exit(ERR_INVALID_ARG);
        }

        printf("Got %s with value %d\n", argName, defaultValue);
        *found = true;
    }

    return defaultValue;
}

void parseArgs(int argc, char** argv){
    int i = 1;
    bool found;
    while(i < argc){
        found = false;
        startModel = getOrDefault(argc, argv, &found, &i, "--model-start", "-ms", true, startModel);
        endModel   = getOrDefault(argc, argv, &found, &i, "--model-end",   "-me", true, endModel);
        startGrid  = getOrDefault(argc, argv, &found, &i, "--grid-start",  "-gs", true, startGrid);
        endGrid    = getOrDefault(argc, argv, &found, &i, "--grid-end",    "-ge", true, endGrid);
        onlyOne    = getOrDefault(argc, argv, &found, &i, "--only-one",    "-oo", false, onlyOne ? 1 : 0) == 1 ? true : false;

        batchSize           = getOrDefault(argc, argv, &found, &i, "--batch-size",  "-bs", true, batchSize);
        slaveBatchSize      = getOrDefault(argc, argv, &found, &i, "--slave-batch-size",  "-sbs", true, slaveBatchSize);
        computeBatchSize    = getOrDefault(argc, argv, &found, &i, "--compute-batch-size",  "-cbs", true, computeBatchSize);
        cpuComputeBatchSize = getOrDefault(argc, argv, &found, &i, "--cpu-compute-batch-size",  "-ccbs", true, cpuComputeBatchSize);

        threadBalancing        = getOrDefault(argc, argv, &found, &i, "--thread-balancing", "-tb", false, threadBalancing ? 1 : 0) == 1 ? true : false;
        slaveBalancing         = getOrDefault(argc, argv, &found, &i, "--slave-balancing", "-sb", false, slaveBalancing ? 1 : 0) == 1 ? true : false;
        slaveDynamicScheduling = getOrDefault(argc, argv, &found, &i, "--slave-dynamic-balancing", "-sdb", false, slaveDynamicScheduling ? 1 : 0) == 1 ? true : false;
        cpuDynamicScheduling   = getOrDefault(argc, argv, &found, &i, "--cpu-dynamic-balancing", "-cdb", false, cpuDynamicScheduling ? 1 : 0) == 1 ? true : false;
        threadBalancingAverage = getOrDefault(argc, argv, &found, &i, "--thread-balancing-avg", "-tba", false, threadBalancingAverage ? 1 : 0) == 1 ? true : false;

        if (getOrDefault(argc, argv, &found, &i, "--cpu", "-cpu", false, 0) == 1) processingType = PROCESSING_TYPE_CPU;
        if (getOrDefault(argc, argv, &found, &i, "--gpu", "-gpu", false, 0) == 1) processingType = PROCESSING_TYPE_GPU;
        if (getOrDefault(argc, argv, &found, &i, "--both", "-both", false, 0) == 1) processingType = PROCESSING_TYPE_BOTH;

        if(!found && i < argc){
            printf("Unknown argument: %s\n", argv[i]);
            break;
        }
    }
}

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

    parseArgs(argc, argv);

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
            parameters.processingType = processingType;
            parameters.overrideMemoryRestrictions = true;
            parameters.finalizeAfterExecution = false;
            parameters.printProgress = false;
            parameters.benchmark = false;

            parameters.dataPtr = (void*) modelDataPtr;
            parameters.dataSize = (1 + stations*(9 - (m<2 ? 0 : 1))) * sizeof(float);

            parameters.threadBalancing          = threadBalancing;
            parameters.slaveBalancing           = slaveBalancing;
            parameters.slaveDynamicScheduling   = slaveDynamicScheduling;
            parameters.cpuDynamicScheduling     = cpuDynamicScheduling;
            parameters.threadBalancingAverage   = threadBalancingAverage;

            parameters.batchSize                = batchSize;
            parameters.slaveBatchSize           = slaveBatchSize;
            parameters.computeBatchSize         = computeBatchSize;
            parameters.cpuComputeBatchSize      = cpuComputeBatchSize;

            parameters.blockSize                = blockSize;
            parameters.gpuStreams               = gpuStreams;

            parameters.slowStartLimit           = slowStartLimit;
            parameters.slowStartBase            = slowStartBase;
            parameters.minMsForRatioAdjustment  = minMsForRatioAdjustment;

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
