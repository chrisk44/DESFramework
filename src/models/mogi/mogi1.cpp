#include "mogi.h"
#include "mogi_common.h"

namespace mogi {

__host__ __device__ float doValidateM1(double* x, void* dataPtr){
    float *xp, *yp, *zp, *de, *dn, *dv, *se, *sn, *sv;
    float r1, ux1, uy1, uz1, dux, duy, duz;
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

    r1 = powf((3.0 * fabsf(x[3])) / (4.0 * PI), 1.0 / 3.0);

    bp = 0;
	if ((r1 / x[2]) <= 0.4) {
		bp = 1;
		for (m = 0; m < stations; m++) {
			ux1 = f(x[0], x[1], x[2], x[3], xp[m], yp[m], zp[m]) * sinf(t(x[0], x[1], xp[m], yp[m]));
			dux = fabsf(ux1  - de[m]);

			if (dux > se[m]) {
				bp = 0;
				break;
			}

			uy1 = f(x[0], x[1], x[2], x[3], xp[m], yp[m], zp[m]) * cosf(t(x[0], x[1], xp[m], yp[m]));
			duy = fabsf(uy1 - dn[m]);

			if (duy > sn[m]) {
				bp = 0;
				break;
			}

			uz1 = h(x[0], x[1], x[2], x[3], xp[m], yp[m], zp[m]);
			duz = fabsf(uz1  - dv[m]);

			if (duz > sv[m]) {
				bp = 0;
				break;
			}
		}
	}

    return bp;
}

}
