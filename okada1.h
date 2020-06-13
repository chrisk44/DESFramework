#include "framework.h"
#include "okada_common.h"

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
        float *xp, *yp, *de, *dn, *dv, *se, *sn, *sv;
        float stro1, dipo1, rakeo1, do1, nc1, ec1, a1, b1, x1, y1, p1, q1, ue1, un1, ux1, uy1, uz1, U11, U21, U31;
	    float dux, duy, duz;
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
    	de = &displacements[2 * stations];
    	dn = &displacements[3 * stations];
    	dv = &displacements[4 * stations];
    	se = &displacements[5 * stations];
    	sn = &displacements[6 * stations];
    	sv = &displacements[7 * stations];

        bp = 0;
        if (((x[ 5] / x[ 6]) > 1.0) && ((x[ 5] / x[ 6]) < 4.0)) {
            bp = 1;
            for (m = 0; m < stations; m++) {
                stro1  = x[ 3] * PI / 180.0;
                dipo1  = x[ 4] * PI / 180.0;
                rakeo1 = x[ 7] * PI / 180.0;
                U11    = cosf(rakeo1) * x[ 8];
                U21    = sinf(rakeo1) * x[ 8];
                U31    = x[ 9];

                /*
                 * Converts fault coordinates (E,N,DEPTH) relative to centroid
                 * into Okada's reference system (X,Y,D)
                 */
                do1 = x[ 2] + sinf(dipo1) * x[ 6];
                ec1 = xp[m] - x[ 0] + cosf(stro1) * cosf(dipo1) * x[ 6] / 2.0;
                nc1 = yp[m] - x[ 1] - sinf(stro1) * cosf(dipo1) * x[ 6] / 2.0;
                x1  = cosf(stro1) * nc1 + sinf(stro1) * ec1 + x[ 5] / 2.0;
                y1  = sinf(stro1) * nc1 - cosf(stro1) * ec1 + cosf(dipo1) * x[ 6];

                /*
                 * Variable substitution (independent from xi and eta)
                 */
                p1 = y1 * cosf(dipo1) + do1 * sinf(dipo1);
                q1 = y1 * sinf(dipo1) - do1 * cosf(dipo1);

                /*
                 * displacements
                 */
                a1 = p1 - x[ 6];
                b1 = x1 - x[ 5];

                /*
                 * displacements
                 */
                ue1 = -U11/(2.0*PI)*(ux_ss(x1,p1,q1,dipo1,NU)-ux_ss(x1,a1,q1,dipo1,NU)-ux_ss(b1,p1,q1,dipo1,NU)+ux_ss(b1,a1,q1,dipo1,NU)) -
                       U21/(2.0*PI)*(ux_ds(x1,p1,q1,dipo1,NU)-ux_ds(x1,a1,q1,dipo1,NU)-ux_ds(b1,p1,q1,dipo1,NU)+ux_ds(b1,a1,q1,dipo1,NU)) +
                       U31/(2.0*PI)*(ux_tf(x1,p1,q1,dipo1,NU)-ux_tf(x1,a1,q1,dipo1,NU)-ux_tf(b1,p1,q1,dipo1,NU)+ux_tf(b1,a1,q1,dipo1,NU));

                un1 = -U11/(2.0*PI)*(uy_ss(x1,p1,q1,dipo1,NU)-uy_ss(x1,a1,q1,dipo1,NU)-uy_ss(b1,p1,q1,dipo1,NU)+uy_ss(b1,a1,q1,dipo1,NU)) -
                       U21/(2.0*PI)*(uy_ds(x1,p1,q1,dipo1,NU)-uy_ds(x1,a1,q1,dipo1,NU)-uy_ds(b1,p1,q1,dipo1,NU)+uy_ds(b1,a1,q1,dipo1,NU)) +
                       U31/(2.0*PI)*(uy_tf(x1,p1,q1,dipo1,NU)-uy_tf(x1,a1,q1,dipo1,NU)-uy_tf(b1,p1,q1,dipo1,NU)+uy_tf(b1,a1,q1,dipo1,NU));

                ux1 = sinf(stro1) * ue1 - cosf(stro1) * un1;
                uy1 = cosf(stro1) * ue1 + sinf(stro1) * un1;

                uz1 = -U11/(2.0*PI)*(uz_ss(x1,p1,q1,dipo1,NU)-uz_ss(x1,a1,q1,dipo1,NU)-uz_ss(b1,p1,q1,dipo1,NU)+uz_ss(b1,a1,q1,dipo1,NU)) -
                       U21/(2.0*PI)*(uz_ds(x1,p1,q1,dipo1,NU)-uz_ds(x1,a1,q1,dipo1,NU)-uz_ds(b1,p1,q1,dipo1,NU)+uz_ds(b1,a1,q1,dipo1,NU)) +
                       U31/(2.0*PI)*(uz_tf(x1,p1,q1,dipo1,NU)-uz_tf(x1,a1,q1,dipo1,NU)-uz_tf(b1,p1,q1,dipo1,NU)+uz_tf(b1,a1,q1,dipo1,NU));

                dux = fabsf(ux1 - de[m]);
                duy = fabsf(uy1 - dn[m]);
                duz = fabsf(uz1 - dv[m]);

                if ((dux > se[m]) || (duy > sn[m]) || (duz > sv[m])) {
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
    parameters.D = 10;
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
    Limit limits[10];
    limits[0] = Limit { 0, 1, 50000 };
    limits[1] = Limit { 0, 1, 50000 };
    limits[2] = Limit { 0, 1, 50000 };
    limits[3] = Limit { 0, 1, 50000 };
    limits[4] = Limit { 0, 1, 50000 };
    limits[5] = Limit { 0, 1, 50000 };
    limits[6] = Limit { 0, 1, 50000 };
    limits[7] = Limit { 0, 1, 50000 };
    limits[8] = Limit { 0, 1, 50000 };
    limits[9] = Limit { 0, 1, 50000 };

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
