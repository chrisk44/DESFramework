#include <iostream>
#include <cstdlib>
#include <fstream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstring>

#include "des/desf.h"
#include "des/scheduler/constantScheduler.h"
#include "des/scheduler/hrdlsScheduler.h"
#include "des/scheduler/hrslsScheduler.h"
#include "des/scheduler/uniformScheduler.h"

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

std::string dataPath = "../data";
bool onlyOne = false;
int startModel = 0;
int endModel = 3;
int startGrid = 1;
int endGrid = 6;

desf::ProcessingType processingType = desf::PROCESSING_TYPE_GPU;

std::string interNodeScheduler = "HRDLS,bs=1000000,minTime=10,ss=6,ssb=50000,latest";
std::string intraNodeScheduler = "HRDLS,bs=100000,minTime=10,ss=3,ssb=50000,latest";
// Uniform
// Constant,k=<constant batch size>
// HRSLS,minTime=<>,minElements=<>,average,latest
// HRDLS,bs=<>,ss=<>,ssbase=<>,minTime=<>,minElements=<>,average,latest

unsigned long cpuComputeBatchSize = 1e+04;
bool cpuDynamicScheduling         = true;

unsigned long computeBatchSize = 20;
int blockSize                  = 1024;
int gpuStreams                 = 8;

bool debug = false;

template<typename T> T fromString(char* str);
template<> float fromString<float>(char* str){ return atof(str); }
template<> int fromString<int>(char* str){ return atoi(str); }
template<> unsigned long fromString<unsigned long>(char* str){ return atoi(str); }
template<> bool fromString<bool>(char* str){ return strcmp(str, "1")==0 || strcmp(str, "true") == 0; }
template<> std::string fromString<std::string>(char* str){ return std::string(str); }

template<typename T>
T getOrDefault(int argc, char** argv, bool* found, int* i, const char* argName, const char* argNameShort, bool hasArgument, T defaultValue){
    if(*i >= argc)
        return defaultValue;

    if(strcmp(argv[*i], argName) == 0 || strcmp(argv[*i], argNameShort) == 0){
        // If it has an argument and we have it...
        if(hasArgument && (*i + 1) < argc){
            defaultValue = fromString<T>(argv[*i+1]);
            *i += 2;
        }
        // else if it doesn't have a second argument, so just mark it as 'found'
        else if(!hasArgument){
            defaultValue = {1};
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

std::vector<std::string> splitString(std::string str, std::string delimiter) {
    std::vector<std::string> result;

    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        result.push_back(str.substr(0, pos));
        str.erase(0, pos + delimiter.length());
    }
    result.push_back(str);

    return result;
}

size_t getNumberOrProduct(std::string str, size_t multiplier) {
    if(str[0] == '*') {
        return multiplier * atof(str.substr(1).c_str());
    }

    return atoll(str.c_str());
}

template<typename node_id_t>
desf::Scheduler<node_id_t>* getSchedulerFromString(std::string str, size_t totalElements) {
    auto parts = splitString(str, ",");
    if(parts.size() == 0)
        throw std::invalid_argument("Malformed scheduler string: " + str);

    if(parts[0] == "Uniform") {
        if(parts.size() > 1)
            throw std::invalid_argument("Uniform scheduler doesn't take any arguments");

        return new desf::UniformScheduler<node_id_t>();
    } else if(parts[0] == "Constant") {
        if(parts.size() != 2)
            throw std::invalid_argument("Constant scheduler requires one argument. Try \"Constant,10000\".");

        return new desf::ConstantScheduler<node_id_t>(getNumberOrProduct(parts[1].c_str(), totalElements));
    } else if(parts[0] == "HRSLS" || parts[0] == "HRDLS") {
        if(parts[0] == "HRDLS" && parts.size() < 2)
            throw std::invalid_argument("HRDLS requires at least a constant batch size");

        size_t batchSize = 0;
        float minTime = 0.f;
        size_t minElements = 0;
        int ss = 0;
        size_t ssb = 0;
        bool average = false;

        for(size_t i=1; i<parts.size(); i++) {
            auto pair = splitString(parts[i], "=");
            if(!(pair.size() == 2 || (pair.size() == 1 && (pair[0] == "average" || pair[0] == "latest"))))
                throw std::invalid_argument("Malformed pair: " + parts[i]);

            if(pair[0] == "average") average = true;
            else if(pair[0] == "latest") average = false;
            else if(pair[0] == "bs") batchSize = getNumberOrProduct(pair[1].c_str(), totalElements);
            else if(pair[0] == "minTime") minTime = atof(pair[1].c_str());
            else if(pair[0] == "minElements") minElements = getNumberOrProduct(pair[1].c_str(), totalElements);
            else if(pair[0] == "ss") ss = atoi(pair[1].c_str());
            else if(pair[0] == "ssb") ssb = getNumberOrProduct(pair[1].c_str(), totalElements);
            else throw std::invalid_argument("Unknown argument: " + pair[0]);
        }

        desf::HRSLSScheduler<node_id_t> *scheduler;
        if(parts[0] == "HRDLS") {
            auto hrdls = new desf::HRDLSScheduler<node_id_t>(batchSize, totalElements);
            hrdls->setSlowStart(ss, ssb);
            scheduler = hrdls;
        } else {
            scheduler = new desf::HRSLSScheduler<node_id_t>(totalElements);
        }

        scheduler->setMinTimeToAdjustRatios(minTime);
        scheduler->setMinElementsToAdjustRatios(minElements);
        if(average) scheduler->setUseAverageRatio();
        else scheduler->setUseLatestRatio();

        return scheduler;
    } else {
        throw std::invalid_argument("Unknown scheduler: " + parts[0]);
    }
}

void printHelp(){
    printf(
        "DES Framework Usage:\n"
        "To run locally:    mpirun -n 2 ./parallelFramework <options>\n"
        "To run on cluster: mpirun --host localhost,localhost,remotehost1,remotehost2 ~/DESFramework/parallelFramework <options>\n\n"
        "Available options (every option takes a number as an argument. For true-false arguments use 0 or 1. -cpu, -gpu, -both don't require an argument.):\n"
        "Model/Grid selection (must be the same for every participating system):\n"
        "--data                     -d              The directory containing the data files\n"
        "--model-start              -ms             The first model to test (1-4).\n"
        "--model-end                -me             The last model to test (1-4).\n"
        "--grid-start               -gs             The first grid to test (1-6).\n"
        "--grid-end                 -ge             The last grid to test (1-6).\n"
        "--only-one                 -oo             Do only one run for each grid regardless of the time it takes.\n"
        "\n"
        "Load balancing (--thread-balancing must be the same for every system, the rest can be freely adjusted per system):\n"
        "--internode-scheduler      -inter          The scheduler that will be used in the Inter-node scheduler\n"
        "--intranode-scheduler      -intra          The scheduler that will be used in the Intra-node scheduler\n"
        "A scheduler can be defined as:\n"
        " Uniform\n"
        " Constant,k=<constant batch size>\n"
        " HRSLS,minTime=<>,minElements=<>,average,latest\n"
        " HRDLS,bs=<>,ss=<>,ssbase=<>,minTime=<>,minElements=<>,average,latest\n"
        "\n"
        " ** Any number referencing a count of elements can also be defined as \"*0.2\" where the given number will be multiplied by the total elements of the grid\n"
        "\n"
        "CPU parameters (can be defined separately for each slave):\n"
        "--cpu-compute-batch-size   -ccbs           The batch size for CPU dynamic scheduling.\n"
        "--cpu-dynamic-balancing    -cdb            Enables dynamic scheduling in the compute thread level for the CPU worker thread. When enabled, the elements\n"
        "                                           that have been assigned to the CPU worker thread will be assigned dynamically to each CPU core using a CPU batch size,\n"
        "                                           as opposed to statically assigning the elements equally to the available cores.\n"
        "\n"
        "GPU parameters (can be defined separately for each slave):\n"
        "--compute-batch-size       -cbs            The number of elements that each GPU thread will compute.\n"
        "--block-size               -bls            The number of threads in each GPU block.\n"
        "--gpu-streams              -gs             The number of GPU streams to be used to dispatch work to the GPU.\n"
        "\n"
        "Resource selection (can be defined separately for each slave) (these don't require arguments, obviously use only one of them):\n"
        "--cpu                      -cpu            Use only the CPUs of the system\n"
        "--gpu                      -gpu            Use only the GPUs of the system\n"
        "--both                     -both           Use all CPUs and GPUs of the system\n"
        "\n"
        "--debug                    -dbg            Wait for debugger to attach\n"
    );
}

void parseArgs(int argc, char** argv){
    int i = 1;
    bool found;
    while(i < argc){
        found = false;
        dataPath   = getOrDefault(argc, argv, &found, &i, "--data",         "-d", true, dataPath);
        startModel = getOrDefault(argc, argv, &found, &i, "--model-start", "-ms", true, startModel + 1) - 1;
        endModel   = getOrDefault(argc, argv, &found, &i, "--model-end",   "-me", true, endModel + 1) - 1;
        startGrid  = getOrDefault(argc, argv, &found, &i, "--grid-start",  "-gs", true, startGrid);
        endGrid    = getOrDefault(argc, argv, &found, &i, "--grid-end",    "-ge", true, endGrid);
        onlyOne    = getOrDefault(argc, argv, &found, &i, "--only-one",    "-oo", false, onlyOne ? 1 : 0) == 1 ? true : false;

        interNodeScheduler = getOrDefault(argc, argv, &found, &i, "--internode-scheduler", "-inter", true, interNodeScheduler);
        intraNodeScheduler = getOrDefault(argc, argv, &found, &i, "--intranode-scheduler", "-intra", true, intraNodeScheduler);

        cpuComputeBatchSize  = getOrDefault(argc, argv, &found, &i, "--cpu-compute-batch-size",  "-ccbs", true, cpuComputeBatchSize);
        cpuDynamicScheduling   = getOrDefault(argc, argv, &found, &i, "--cpu-dynamic-balancing", "-cdb", true, cpuDynamicScheduling ? 1 : 0) == 1 ? true : false;

        computeBatchSize     = getOrDefault(argc, argv, &found, &i, "--compute-batch-size",  "-cbs", true, computeBatchSize);
        blockSize  = getOrDefault(argc, argv, &found, &i, "--block-size",  "-bls", true, blockSize);
        gpuStreams = getOrDefault(argc, argv, &found, &i, "--gpu-streams", "-gstr", true, gpuStreams);

        if (getOrDefault(argc, argv, &found, &i, "--cpu", "-cpu", false, false) == 1) processingType = desf::PROCESSING_TYPE_CPU;
        if (getOrDefault(argc, argv, &found, &i, "--gpu", "-gpu", false, false) == 1) processingType = desf::PROCESSING_TYPE_GPU;
        if (getOrDefault(argc, argv, &found, &i, "--both", "-both", false, false) == 1) processingType = desf::PROCESSING_TYPE_BOTH;

        debug = getOrDefault(argc, argv, &found, &i, "--debug", "-dbg", false, false);

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

    if(dataPath.size() == 0 || dataPath.back() != '/')
        dataPath.append("/");
}

int main(int argc, char** argv){
    const std::string modelNames[4] = {
        "mogi1",
        "mogi2",
        "okada1",
        "okada2"
    };

    // Scale factor, must be >0
    float k = 2;

    int rank, commSize;

    float finalResults[4][6];
    for(int i=0; i<4; i++)
        for(int j=0; j<6; j++)
            finalResults[i][j] = 0.0;

    parseArgs(argc, argv);

    // Initialize MPI manually
    printf("Initializing MPI\n");
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool isMaster = rank == 0;

    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    if(debug) {
        volatile int i = 0;
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        printf("[%d] PID %d on %s ready to attach\n", rank, getpid(), hostname);
        fflush(stdout);
        sleep(20);
    }

    const std::vector<desf::validationFunc_t> cpuForwardModels = {
        validate_cpuM1,
        validate_cpuM2,
        validate_cpuO1,
        validate_cpuO2
    };
    const desf::toBool_t cpuObjective = toBool_cpu;

    std::vector<std::map<int, desf::validationFunc_t>> gpuForwardModels;
    std::map<int, desf::toBool_t> gpuObjective;
    if(!isMaster){
        gpuForwardModels = {
            desf::DesFramework::getGpuPointersFromSymbol<desf::validationFunc_t, validate_gpuM1>(),
            desf::DesFramework::getGpuPointersFromSymbol<desf::validationFunc_t, validate_gpuM2>(),
            desf::DesFramework::getGpuPointersFromSymbol<desf::validationFunc_t, validate_gpuO1>(),
            desf::DesFramework::getGpuPointersFromSymbol<desf::validationFunc_t, validate_gpuO2>()
        };
        gpuObjective = desf::DesFramework::getGpuPointersFromSymbol<desf::toBool_t, toBool_gpu>();
    } // else do not attempt to retrieve addresses

    // For each model...
    for(int m=startModel; m<=endModel; m++){
        if(isMaster) printf("[%d] Starting model %d/4...\n", rank, m+1);
        // Open displacements file
        std::string displFilename = dataPath + modelNames[m] + "/displ.txt";
        std::ifstream dispfile(displFilename, std::ios::in);

        // Count stations
        int stations = 0;
        {
            std::string tmp;
            while(getline(dispfile, tmp)) stations++;
        }

        if(stations < 1){
            printf("[%d] [%s \\ N/A] Got 0 displacements. Exiting.\n", rank, modelNames[m].c_str());
            exit(2);
        }

        // Reset the file
        dispfile.close();
        dispfile.open(displFilename, std::ios::in);

        // Create the model's parameters struct (the model's input data)
        float modelDataPtr[1 + stations * (9 - (m<2 ? 0 : 1))];
        modelDataPtr[0] = (float) stations;

        // Read each station's displacement data
        float *dispPtr = &modelDataPtr[1];
        if(m < 2){
            // Mogi models have x,y,z,...
            int i = 0;
            float x, y, z, de, dn, dv, se, sn, sv;
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
            int i = 0;
            float x, y, de, dn, dv, se, sn, sv;
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
        for(int g=startGrid; g<=endGrid; g++){
            if(isMaster) printf("[%d] Starting grid %d/6\n", rank, g);
            if(m == 3 && g > 4)
                continue;

            // Open grid file
            std::string gridFilename = dataPath + modelNames[m] + "/grid" + std::to_string(g) + ".txt";
            std::ifstream gridfile(gridFilename, std::ios::in);

            desf::DesConfig config(false);

            // Read each dimension's grid information
            unsigned long totalElements = 1;
            {
                int i = 0;
                float low, high, step;
                while(gridfile >> low >> high >> step){
                    // Create the limit (lower is inclusive, upper is exclusive)
                    high += step;
                    config.limits.push_back(desf::Limit{ low, high, (unsigned int) ((high-low)/step), step });
                    totalElements *= config.limits.back().N;
                    i++;
                }
            }

            // Close the file
            gridfile.close();

            // Set up the framework's parameters struct
            config.model.D = config.limits.size();
            config.resultSaveType = desf::SAVE_TYPE_LIST;
            config.processingType = processingType;
            config.output.overrideMemoryRestrictions = true;
            config.handleMPI = false;
            config.printProgress = false;
            config.benchmark = false;

            config.model.dataPtr = (void*) modelDataPtr;
            config.model.dataSize = (1 + stations*(9 - (m<2 ? 0 : 1))) * sizeof(float);

            config.interNodeScheduler = getSchedulerFromString<int>(interNodeScheduler, totalElements);
            config.intraNodeScheduler = getSchedulerFromString<desf::ComputeThreadID>(intraNodeScheduler, totalElements);

            if(!isMaster) {
                config.cpu.dynamicScheduling    = cpuDynamicScheduling;
                config.cpu.computeBatchSize     = cpuComputeBatchSize;
                config.cpu.forwardModel = cpuForwardModels[m];
                config.cpu.objective = cpuObjective;

                for(const auto& pair : gpuObjective){
                    desf::GpuConfig gpuConfig;
                    gpuConfig.computeBatchSize = computeBatchSize;
                    gpuConfig.blockSize        = blockSize;
                    gpuConfig.streams          = gpuStreams;
                    gpuConfig.forwardModel     = gpuForwardModels[m][pair.first];
                    gpuConfig.objective        = pair.second;
                    config.gpu.insert(std::make_pair(pair.first, gpuConfig));
                }
            }

            float totalTime = 0;        //msec
            int numOfRuns = 0;
            int numOfResults = -2;
            // Run at least 10 seconds, and stop after 10 runs or 2 minutes
            while(true){
                // Initialize the framework object
                desf::DesFramework framework(config);

                // Start the computation
                desf::Stopwatch sw;
                sw.start();
                framework.run();
                sw.stop();

                if(isMaster){
                    int size = framework.getList().size();
                    if(size != numOfResults && numOfResults != -2){
                        printf("[%s \\ %d] Number of results from run %d don't match: %d -> %d.\n",
                                        modelNames[m].c_str(), g, numOfRuns, numOfResults, size);
                    }
                    numOfResults = size;
                    // printf("[%s \\ %d] Run %d: %f ms, %d results\n", modelNames[m], g, numOfRuns, size, sw.getMsec());
                }

                totalTime += sw.getMsec();
                numOfRuns++;

                int moveToNext;
                if(isMaster){
                    if(commSize > 2 || processingType == desf::PROCESSING_TYPE_BOTH) printf("\n");
                    if(onlyOne || (totalTime > 10 * 1000 && (numOfRuns >= 10 || totalTime >= 1 * 60 * 1000))){
                        finalResults[m][g] = totalTime/numOfRuns;
                        printf("[%s \\ %d] Time: %f ms in %d runs\n",
                                    modelNames[m].c_str(), g, totalTime/numOfRuns, numOfRuns);

                        std::string outFilename = "results_" + modelNames[m] + "_" + std::to_string(g) + ".txt";
                        // Open file to write results
                        std::ofstream outfile(outFilename, std::ios::out | std::ios::trunc);

                        auto list = framework.getList();
                        printf("[%s \\ %d] Results: %lu\n", modelNames[m].c_str(), g, list.size());
                        for(int i=0; i<std::min((int) list.size(), 5); i++){
                            printf("[%s \\ %d] ( ", modelNames[m].c_str(), g);
                            for(auto v : list[i])
                                printf("%lf ", v);

                            printf(")\n");
                        }
                        if(list.size() > 5)
                            printf("[%s \\ %d] ...%lu more results\n", modelNames[m].c_str(), g, list.size()-5);

                        printf("\n");
                        if(g==6)
                            printf("\n");

                        for(auto point : list){
                            for(auto v : point)
                                outfile << v << " ";

                            outfile << std::endl;
                        }

                        outfile.close();
                        moveToNext = 1;
                    }else{
                        moveToNext = 0;
                    }
                }

                MPI_Bcast(&moveToNext, 1, MPI_INT, 0, MPI_COMM_WORLD);

                if(moveToNext)
                    break;
            } // end while time too short

            delete config.interNodeScheduler;
            delete config.intraNodeScheduler;

            if(onlyOne)
                break;
        } // end for each grid

    }   // end for each model

    if(isMaster){
        printf("Final results:\n");
        for(int m=startModel; m<=endModel; m++){
            for(int g=startGrid; g<=endGrid; g++)
                printf("%f\n", finalResults[m][g]);

            printf("\n");
        }
    }

    MPI_Finalize();

    return 0;
}
