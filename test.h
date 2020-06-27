#include <fstream>

#include "framework.h"
#include "mogi_common.h"

using namespace std;

class MyModel : public Model{
public:
    inline __host__ RESULT_TYPE validate_cpu(DATA_TYPE* x, void* dataPtr){
        return doValidate(x, dataPtr);
    }

    inline __device__ RESULT_TYPE validate_gpu(DATA_TYPE* x, void* dataPtr){
        return doValidate(x, dataPtr);
    }

    inline __host__ __device__ RESULT_TYPE doValidate(DATA_TYPE* x, void* dataPtr){
        return x[0] > 0 && x[1] > 0 ? 1 : 0;
    }

    inline __host__ __device__ bool toBool(RESULT_TYPE result){
        return result != 0;
    }
};

void run(int argc, char** argv){
    ofstream outfile;
    int result, i, j;

    //char saveFile[] = "./customResults.txt";

    ParallelFrameworkParameters parameters;
    Limit limits[2];
    limits[0] = Limit { -1, 0.001, 100 };
    limits[1] = Limit { -10, 0.01, 100 };

    Stopwatch sw;
    int length;
    DATA_TYPE* list;

    // Create the framework's parameters struct
    parameters.D = 2;
    parameters.resultSaveType = SAVE_TYPE_LIST;
    //parameters.saveFile = saveFile;
    parameters.processingType = PROCESSING_TYPE_CPU;
    parameters.threadBalancing = true;
    parameters.slaveBalancing = true;
    parameters.benchmark = false;
    parameters.batchSize = 20147;

    // Initialize the framework object
    ParallelFramework framework = ParallelFramework(limits, parameters);
    if (! framework.isValid()) {
        cout << "Error initializing framework: " << endl;
        exit(-1);
    }

    // Start the computation
    sw.reset();
    sw.start();
    result = framework.run<MyModel>();
    sw.stop();
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
        exit(-1);
    }
    printf("Time: %f ms\n", sw.getMsec());

    // Open file to write results
    outfile.open("results.txt", ios::out | ios::trunc);

    list = framework.getList(&length);
    printf("Results: %d\n", length);
    for(i=0; i<length; i++){
        printf("(");
        //outfile << "(";

        for(j=0; j<parameters.D-1; j++){
            printf("%f ", list[i*parameters.D + j]);
            outfile << list[i*parameters.D + j] << " ";
        }

        printf("%f)\n", list[i*parameters.D + j]);
        outfile << list[i*parameters.D + j] << endl;
    }

    outfile.close();
}
