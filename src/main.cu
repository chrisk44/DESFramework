#include <iostream>
#include <cstdlib>
#include <fstream>
#include <mpi.h>
#include <cuda_runtime.h>

#include "des/framework.h"

/*
 * Each one of these contain a __host__ __device__ doValidate[MO][123] function
*/
#include "models/mogi/mogi.h"
#include "models/okada/okada.h"

__host__   RESULT_TYPE validate_cpuM1(DATA_TYPE* x, void* dataPtr){ return mogi::doValidateM1(x, dataPtr); }
__device__ RESULT_TYPE validate_gpuM1(DATA_TYPE* x, void* dataPtr){ return mogi::doValidateM1(x, dataPtr); }

__host__   RESULT_TYPE validate_cpuM2(DATA_TYPE* x, void* dataPtr){ return mogi::doValidateM2(x, dataPtr); }
__device__ RESULT_TYPE validate_gpuM2(DATA_TYPE* x, void* dataPtr){ return mogi::doValidateM2(x, dataPtr); }

__host__   RESULT_TYPE validate_cpuO1(DATA_TYPE* x, void* dataPtr){ return okada::doValidateO1(x, dataPtr); }
__device__ RESULT_TYPE validate_gpuO1(DATA_TYPE* x, void* dataPtr){ return okada::doValidateO1(x, dataPtr); }

__host__   RESULT_TYPE validate_cpuO2(DATA_TYPE* x, void* dataPtr){ return okada::doValidateO2(x, dataPtr); }
__device__ RESULT_TYPE validate_gpuO2(DATA_TYPE* x, void* dataPtr){ return okada::doValidateO2(x, dataPtr); }

// __host__   RESULT_TYPE validate_cpuO3(DATA_TYPE* x, void* dataPtr){ return okada::doValidateO3(x, dataPtr); }
// __device__ RESULT_TYPE validate_gpuO3(DATA_TYPE* x, void* dataPtr){ return okada::doValidateO3(x, dataPtr); }

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
bool threadBalancingAverage   = false;

unsigned long batchSize           = UINT_MAX;
float batchSizeFactor             = -1;
unsigned long slaveBatchSize      = UINT_MAX;  // 1e+07
float slaveBatchSizeFactor        = -1;
unsigned long computeBatchSize    = 20;
unsigned long cpuComputeBatchSize = 1e+04;

int blockSize  = 1024;
int gpuStreams = 8;

int slowStartLimit          = 6;
unsigned long slowStartBase = 5e+05;
int minMsForRatioAdjustment = 10;

unsigned long getOrDefault(int argc, char** argv, bool* found, int* i, const char* argName, const char* argNameShort, bool hasArgument, unsigned long defaultValue){
    if(*i >= argc)
        return defaultValue;

    if(strcmp(argv[*i], argName) == 0 || strcmp(argv[*i], argNameShort) == 0){
        // If it has an argument and we have it...
        if(hasArgument && (*i + 1) < argc){
            defaultValue = atoi(argv[*i+1]);
            *i += 2;
        }
        // else if it doesn't have a second argument, so just mark it as 'found'
        else if(!hasArgument){
            defaultValue = 1;
            *i += 1;
        }
        // else if it has an argument and we don't have it
        else{
            fprintf(stderr, "[E] %s requires an additional argument\n", argName);
            exit(ERR_INVALID_ARG);
        }

        *found = true;
    }

    return defaultValue;
}

float getOrDefaultF(int argc, char** argv, bool* found, int* i, const char* argName, const char* argNameShort, bool hasArgument, float defaultValue){
    if(*i >= argc)
        return defaultValue;

    if(strcmp(argv[*i], argName) == 0 || strcmp(argv[*i], argNameShort) == 0){
        // If it has an argument and we have it...
        if(hasArgument && (*i + 1) < argc){
            defaultValue = atof(argv[*i+1]);
            *i += 2;
        }
        // else if it doesn't have a second argument, so just mark it as 'found'
        else if(!hasArgument){
            defaultValue = 1;
            *i += 1;
        }
        // else if it has an argument and we don't have it
        else{
            fprintf(stderr, "[E] %s requires an additional argument\n", argName);
            exit(ERR_INVALID_ARG);
        }

        *found = true;
    }

    return defaultValue;
}

void printHelp(){
    printf(
        "DES Framework Usage:\n"\
        "To run locally:    mpirun -n 2 ./parallelFramework <options>\n"
        "To run on cluster: mpirun --host localhost,localhost,remotehost1,remotehost2 ~/DESFramework/parallelFramework <options>\n\n"
        "Available options (every option takes a number as an argument. For true-false arguments use 0 or 1. -cpu, -gpu, -both don't require an argument.):\n"
        "Model/Grid selection (must be the same for every participating system):\n"
        "--model-start              -ms             The first model to test (1-4).\n"
        "--model-end                -me             The last model to test (1-4).\n"
        "--grid-start               -gs             The first grid to test (1-6).\n"
        "--grid-end                 -ge             The last grid to test (1-6).\n"
        "--only-one                 -oo             Do only one run for each grid regardless of the time it takes.\n"
        "\n"
        "Load balancing (--thread-balancing must be the same for every system, the rest can be freely adjusted per system):\n"
        "--thread-balancing         -tb             Enables the use of HPLS in the slave level for each compute thread.\n"
        "                                           This means that for each assignment, the slave will use HPLS to calculate a ratio which will\n"
        "                                           be multiplied by the slave batch size to determine the number of elements that will be assigned to each compute thread.\n"
        "--slave-balancing          -sb             Enables the use of HPLS in the master level for each slave.\n"
        "                                           This means that for each assignment request, the master will use HPLS to calculate a ratio\n"
        "                                           which will be multiplied by the global batch size to determine the number of elements that\n"
        "                                           should be assigned to that slave.\n"
        "--slave-dynamic-balancing  -sdb            Enables dynamic scheduling in the slave level. When enabled, the slave will assign the elements dynamically to the available\n"
        "                                           resources using the slave batch size, as opposed to assigning them all at once\n"
        "--cpu-dynamic-balancing    -cdb            Enables dynamic scheduling in the compute thread level for the CPU worker thread. When enabled, the elements\n"
        "                                           that have been assigned to the CPU worker thread will be assigned dynamically to each CPU core using a CPU batch size,\n"
        "                                           as opposed to statically assigning the elements equally to the available cores.\n"
        "--thread-balancing-avg     -tba            Causes the slave-level HPLS to use the average ratio for each compute thread instead of the latest one\n. Useful when\n"
        "                                           the elements are heavily imbalanced compute-wise.\n"
        "\n"
        "Element assignment (can be defined separately for each slave, except for --batch-size which is used only by the master process):\n"
        "--batch-size               -bs             The maximum number of elements for each assignment from the master node to a slave, and the multiplier of HPLS ratios.\n"
        "--batch-size-factor        -bsf            The maximum number of elements for each assignment from the master node to a slave, and the multiplier of HPLS ratios (multiplier for total elements of grid).\n"
        "--slave-batch-size         -sbs            The maximum number of elements that a slave can assign to a compute thread at a time, and the multiplier of HPLS ratios.\n"
        "--slave-batch-size-factor  -sbsf           The maximum number of elements that a slave can assign to a compute thread at a time, and the multiplier of HPLS ratios (multiplier for total elements of grid).\n"
        "--compute-batch-size       -cbs            The number of elements that each GPU thread will compute.\n"
        "--cpu-compute-batch-size   -ccbs           The batch size for CPU dynamic scheduling.\n"
        "\n"
        "GPU parameters (can be defined separately for each slave):\n"
        "--block-size               -bls            The number of threads in each GPU block.\n"
        "--gpu-streams              -gs             The number of GPU streams to be used to dispatch work to the GPU.\n"
        "\n"
        "Slow-Start technique (used only by the master system, except for minimum time for ratio adjustment which is also used by the slaves and can be freely adjusted per system):\n"
        "--slow-start-limit         -ssl            The number of assignments that should be limited by the slow-start technique, where after each step the limit is doubled.\n"
        "--slow-start-base          -ssb            The initial number of elements for the slow-start technique which will be doubled after each step.\n"
        "--min-ms-ratio             -mmr            The minimum time in milliseconds that will be considered as valid to be used to adjust HPLS ratios.\n"
        "\n"
        "Resource selection (can be defined separately for each slave) (these don't require arguments, obviously use only one of them):\n"
        "--cpu                      -cpu            Use only the CPUs of the system\n"
        "--gpu                      -gpu            Use only the GPUs of the system\n"
        "--both                     -both           Use all CPUs and GPUs of the system\n"
    );
}

void parseArgs(int argc, char** argv){
    int i = 1;
    bool found;
    while(i < argc){
        found = false;
        startModel = getOrDefault(argc, argv, &found, &i, "--model-start", "-ms", true, startModel + 1) - 1;
        endModel   = getOrDefault(argc, argv, &found, &i, "--model-end",   "-me", true, endModel + 1) - 1;
        startGrid  = getOrDefault(argc, argv, &found, &i, "--grid-start",  "-gs", true, startGrid);
        endGrid    = getOrDefault(argc, argv, &found, &i, "--grid-end",    "-ge", true, endGrid);
        onlyOne    = getOrDefault(argc, argv, &found, &i, "--only-one",    "-oo", false, onlyOne ? 1 : 0) == 1 ? true : false;

        batchSize            = getOrDefault(argc, argv, &found, &i, "--batch-size",  "-bs", true, batchSize);
        batchSizeFactor      = getOrDefaultF(argc, argv, &found, &i, "--batch-size-factor", "-bsf", true, batchSizeFactor);
        slaveBatchSize       = getOrDefault(argc, argv, &found, &i, "--slave-batch-size",  "-sbs", true, slaveBatchSize);
        slaveBatchSizeFactor = getOrDefaultF(argc, argv, &found, &i, "--slave-batch-size-factor",  "-sbsf", true, slaveBatchSizeFactor);
        computeBatchSize     = getOrDefault(argc, argv, &found, &i, "--compute-batch-size",  "-cbs", true, computeBatchSize);
        cpuComputeBatchSize  = getOrDefault(argc, argv, &found, &i, "--cpu-compute-batch-size",  "-ccbs", true, cpuComputeBatchSize);

        threadBalancing        = getOrDefault(argc, argv, &found, &i, "--thread-balancing", "-tb", true, threadBalancing ? 1 : 0) == 1 ? true : false;
        slaveBalancing         = getOrDefault(argc, argv, &found, &i, "--slave-balancing", "-sb", true, slaveBalancing ? 1 : 0) == 1 ? true : false;
        slaveDynamicScheduling = getOrDefault(argc, argv, &found, &i, "--slave-dynamic-balancing", "-sdb", true, slaveDynamicScheduling ? 1 : 0) == 1 ? true : false;
        cpuDynamicScheduling   = getOrDefault(argc, argv, &found, &i, "--cpu-dynamic-balancing", "-cdb", true, cpuDynamicScheduling ? 1 : 0) == 1 ? true : false;
        threadBalancingAverage = getOrDefault(argc, argv, &found, &i, "--thread-balancing-avg", "-tba", true, threadBalancingAverage ? 1 : 0) == 1 ? true : false;

        blockSize  = getOrDefault(argc, argv, &found, &i, "--block-size",  "-bls", true, blockSize);
        gpuStreams = getOrDefault(argc, argv, &found, &i, "--gpu-streams",  "-gs", true, gpuStreams);
        slowStartLimit = getOrDefault(argc, argv, &found, &i, "--slow-start-limit",  "-ssl", true, slowStartLimit);
        slowStartBase = getOrDefault(argc, argv, &found, &i, "--slow-start-base",  "-ssb", true, slowStartBase);
        minMsForRatioAdjustment = getOrDefault(argc, argv, &found, &i, "--min-ms-ratio",  "-mmr", true, minMsForRatioAdjustment);

        if (getOrDefault(argc, argv, &found, &i, "--cpu", "-cpu", false, 0) == 1) processingType = PROCESSING_TYPE_CPU;
        if (getOrDefault(argc, argv, &found, &i, "--gpu", "-gpu", false, 0) == 1) processingType = PROCESSING_TYPE_GPU;
        if (getOrDefault(argc, argv, &found, &i, "--both", "-both", false, 0) == 1) processingType = PROCESSING_TYPE_BOTH;

        if (getOrDefault(argc, argv, &found, &i, "--help", "-help", false, 0) == 1){
            printHelp();
            exit(0);
        }

        if(!found && i < argc){
            printf("Unknown argument: %s\n", argv[i]);
            printHelp();
            exit(1);
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
    int stations, dims, i, j, m, g, result, rank, commSize;
    float *modelDataPtr, *dispPtr;
    float x, y, z, de, dn, dv, se, sn, sv;
    float low, high, step;

    Stopwatch sw;
    int length;
    DATA_TYPE* list;

    float finalResults[4][6];
    for(int i=0; i<4; i++)
        for(int j=0; j<6; j++)
            finalResults[i][j] = 0.0;

    parseArgs(argc, argv);

    // Initialize MPI manually
    printf("Initializing MPI\n");
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    isMaster = rank == 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &commSize);

    // For each model...
    for(m=startModel; m<=endModel; m++){
        if(isMaster) printf("[%d] Starting model %d/4...\n", rank, m+1);
        // Open displacements file
        sprintf(displFilename, "../../data/%s/displ.txt", modelNames[m]);
        dispfile.open(displFilename, ios::in);

        // Count stations
        stations = 0;
        while(getline(dispfile, tmp)) stations++;

        if(stations < 1){
            printf("[%d] [%s \\ %d] Got 0 displacements. Exiting.\n", rank, modelNames[m], g);
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
            if(isMaster) printf("[%d] Starting grid %d/6\n", rank, g);
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

            parameters.batchSize                = batchSizeFactor > 0 ? totalElements * batchSizeFactor : batchSize;
            parameters.slaveBatchSize           = slaveBatchSizeFactor > 0 ? totalElements * slaveBatchSizeFactor : slaveBatchSize;
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
                    printf("[%d] [%s \\ %d] Error initializing framework\n", rank, modelNames[m], g);
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
                    printf("[%d] [%s \\ %d] Error running the computation: %d\n", rank, modelNames[m], g, result);
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
                    if(commSize > 2 || processingType == PROCESSING_TYPE_BOTH) printf("\n");
                    if(onlyOne || (totalTime > 10 * 1000 && (numOfRuns >= 10 || totalTime >= 1 * 60 * 1000))){
                        finalResults[m][g] = totalTime/numOfRuns;
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
            } // end while time too short

            if(onlyOne)
                break;
        } // end for each grid

        delete [] modelDataPtr;
    }   // end for each model

    if(isMaster){
        printf("Final results:\n");
        for(m=startModel; m<=endModel; m++){
            for(g=startGrid; g<=endGrid; g++)
                printf("%f\n", finalResults[m][g]);

            printf("\n");
        }
    }

    MPI_Finalize();

    return 0;
}
