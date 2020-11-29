#include "okada_common.h"

using namespace std;

__host__ __device__ RESULT_TYPE doValidateO1(DATA_TYPE* x, void* dataPtr){
    float *xp, *yp, *de, *dn, *dv, *se, *sn, *sv;
    float stro1, dipo1, rakeo1, do1, nc1, ec1, a1, b1, x1, y1, p1, q1, ue1, un1, ux1, uy1, uz1, U11, U21, U31;
    float dux, duy, duz;
    unsigned long long bp, m;

    // Data
    unsigned long long stations;
    float *displacements;

    stations = ((float*) dataPtr)[0];
    displacements = &(((float*) dataPtr)[1]);

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
                return 0;
            }
        }
    }

    return bp;
}
