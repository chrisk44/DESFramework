#include "framework.h"
#include "mogi_common.h"

using namespace std;

struct MogiParameters{
    unsigned long long stations;
    float displacements[10];
};

class MyModel : public Model{
public:
    __host__ RESULT_TYPE validate_cpu(DATA_TYPE* x, void* dataPtr){
        return doValidate(x, dataPtr);
    }

    __device__ RESULT_TYPE validate_gpu(DATA_TYPE* x, void* dataPtr){
        return doValidate(x, dataPtr);
    }

    __host__ __device__ RESULT_TYPE doValidate(DATA_TYPE* x, void* dataPtr){
        float *xp, *yp, *zp, *de, *dn, *dv, *se, *sn, *sv;
        float dist, r1, r2, ux, ux1, ux2, uy, uy1, uy2, uz, uz1, uz2, dux, duy, duz;
        unsigned long long bp, m;

        // Data
        unsigned long long stations;
        float *displacements;

        // TBD: If the length of 'displacements' is known at compile time...
        stations = ((struct MogiParameters*) dataPtr)->stations;
        displacements = ((struct MogiParameters*) dataPtr)->displacements;

        // // TBD: If the length of 'displacements is not known at compile time'...
        // stations = ((float*) dataPtr)[0];
        // displacements = &(((float*) dataPtr)[1]);

        xp = &displacements[0 * stations];
    	yp = &displacements[1 * stations];
    	zp = &displacements[2 * stations];
    	de = &displacements[3 * stations];
    	dn = &displacements[4 * stations];
    	dv = &displacements[5 * stations];
    	se = &displacements[6 * stations];
    	sn = &displacements[7 * stations];
    	sv = &displacements[8 * stations];

        r1   = powf((3.0 * fabsf(x[3])) / (4.0 * PI), 1.0 / 3.0);
		r2   = powf((3.0 * fabsf(x[7])) / (4.0 * PI), 1.0 / 3.0);
		dist = sqrtf((x[4] - x[0]) * (x[4] - x[0]) + (x[5] - x[1]) * (x[5] - x[1]));

        bp = 0;
		if (((r1 / x[2]) <= 0.4) && ((r2 / x[6]) <= 0.4) && (dist >= 4.0 * fmaxf(r1, r2))) {
			bp = 1;
			for (m = 0; m < stations; m++) {
				ux1 = f(x[0], x[1], x[2], x[3], xp[m], yp[m], zp[m]) * sinf(t(x[0], x[1], xp[m], yp[m]));
				ux2 = f(x[4], x[5], x[6], x[7], xp[m], yp[m], zp[m]) * sinf(t(x[4], x[5], xp[m], yp[m]));
				ux  = ux1 + ux2;
				dux = fabsf(ux - de[m]);

				if (dux > se[m]) {
					bp = 0;
					break;
				}

				uy1 = f(x[0], x[1], x[2], x[3], xp[m], yp[m], zp[m]) * cosf(t(x[0], x[1], xp[m], yp[m]));
				uy2 = f(x[4], x[5], x[6], x[7], xp[m], yp[m], zp[m]) * cosf(t(x[4], x[5], xp[m], yp[m]));
				uy  = uy1 + uy2;
				duy = fabsf(uy - dn[m]);

				if (duy > sn[m]) {
					bp = 0;
					break;
				}

				uz1 = h(x[0], x[1], x[2], x[3], xp[m], yp[m], zp[m]);
				uz2 = h(x[4], x[5], x[6], x[7], xp[m], yp[m], zp[m]);
				uz  = uz1 + uz2;
				duz = fabsf(uz - dv[m]);

				if (duz > sv[m]) {
					bp = 0;
					break;
				}
			}
		}

        return bp;
    }

    bool toBool(RESULT_TYPE result){
        return result != 0;
    }
};

void run(){
    // Initialize framework
    int result = 0;

    // Create the model's parameters struct (the model's input data)
    // TBD: If the length of 'displacements' is known at compile time...
    MogiParameters mogiParameters;
    int displacementsLength = 10;
    mogiParameters.stations = 1;
    for(int i=0; i<displacementsLength; i++){
        mogiParameters.displacements[i] = 0;
    }

    // // TBD: If the length of 'displacements is not known at compile time'...
    // int displacementsLength = 10;        // <-- Determine this at runtime
    // int stations = 2;                    // <-- Determine this at runtime
    // float* modelDataPtr = (float*) malloc((displacementsLength+1) * sizeof(float));
    // modelDataPtr[0] = stations;
    // for(int i=0; i<displacementsLength; i++){
    //     modelDataPtr[i+1] = 0;           // <-- Write at index i+1 because the first elements of the array is reserved for 'stations'
    // }

    // Create the framework's parameters struct
    ParallelFrameworkParameters parameters;
    parameters.D = 8;
    parameters.resultSaveType = SAVE_TYPE_LIST;
    parameters.processingType = PROCESSING_TYPE_BOTH;
    parameters.dataPtr = &mogiParameters;
    parameters.dataSize = sizeof(mogiParameters);
    // parameters.dataPtr = (void*) modelDataPtr;
    // parameters.dataSize = (displacementsLength+1) * sizeof(float);
    parameters.threadBalancing = true;
    parameters.slaveBalancing = true;
    parameters.benchmark = false;
    parameters.batchSize = 20000000;

    // Create the limits for each dimension (lower is inclusive, upper is exclusive)
    Limit limits[8];
    limits[0] = Limit { 0, 1, 50000 };
    limits[1] = Limit { 0, 1, 50000 };
    limits[2] = Limit { 0, 1, 50000 };
    limits[3] = Limit { 0, 1, 50000 };
    limits[4] = Limit { 0, 1, 50000 };
    limits[5] = Limit { 0, 1, 50000 };
    limits[6] = Limit { 0, 1, 50000 };
    limits[7] = Limit { 0, 1, 50000 };

    // Initialize the framework object
    ParallelFramework framework = ParallelFramework(limits, parameters);
    if (! framework.isValid()) {
        cout << "Error initializing framework: " << result << endl;
        exit(-1);
    }

    // Start the computation
    Stopwatch sw;
    sw.reset();
    sw.start();
    result = framework.run<MyModel>();
    sw.stop();
    if (result != 0) {
        cout << "Error running the computation: " << result << endl;
        exit(-1);
    }
    printf("Time: %f ms\n", sw.getMsec());

    int length;
    DATA_TYPE* list = framework.getList(&length);
    printf("Results:\n");
    for(int k=0; k<length; k++){
        printf("( ");
        for(int i=0; i<parameters.D; i++)
            printf("%f ", list[i*k]);
        printf(")\n");
    }
}
