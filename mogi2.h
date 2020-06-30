#include <fstream>

#include "framework.h"
#include "mogi_common.h"

using namespace std;

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

        stations = ((float*) dataPtr)[0];
        displacements = &(((float*) dataPtr)[1]);

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

    inline __host__ __device__ bool toBool(RESULT_TYPE result){
        return result != 0;
    }
};

void run(int argc, char** argv){
    ifstream dispfile, gridfile;
    ofstream outfile;
    string tmp;
    int stations, dims, i, j, result;
    float *modelDataPtr, *dispPtr;
    float x, y, z, de, dn, dv, se, sn, sv, k;
    float low, high, step;

    ParallelFrameworkParameters parameters;
    Limit limits[8];

    Stopwatch sw;
    int length;
    DATA_TYPE* list;

    if(argc != 4){
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
    if(dims != 8){
        printf("Got %d dimensions, expected 8. Exiting.\n", dims);
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
        limits[i] = Limit{ low, high, (unsigned long) ((high-low)/step) };
        i++;
    }

    dispfile.close();
    gridfile.close();

    // Create the framework's parameters struct
    parameters.D = 8;
    parameters.resultSaveType = SAVE_TYPE_LIST;
    parameters.processingType = PROCESSING_TYPE_BOTH;
    parameters.dataPtr = (void*) modelDataPtr;
    parameters.dataSize = (1 + stations*9) * sizeof(float);
    parameters.threadBalancing = true;
    parameters.slaveBalancing = true;
    parameters.benchmark = false;
    parameters.batchSize = 200000000;
    parameters.computeBatchSize = 1000;
    parameters.blockSize = 256;
    parameters.gpuStreams = 8;

    // Initialize the framework object
    ParallelFramework framework = ParallelFramework(limits, parameters);
    if (! framework.isValid()) {
        cout << "Error initializing framework: " << endl;
        exit(-1);
    }

    // Start the computation
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
        // printf("(");

        for(j=0; j<parameters.D-1; j++){
            // printf("%f ", list[i*parameters.D + j]);
            outfile << list[i*parameters.D + j] << " ";
        }

        // printf("%f)\n", list[i*parameters.D + j]);
        outfile << list[i*parameters.D + j] << endl;
    }

    outfile.close();

    delete [] modelDataPtr;
}
