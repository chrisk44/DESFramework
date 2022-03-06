#include "okada.h"
#include "okada_common.h"

namespace okada {

__host__ __device__ float doValidateO3(double* x, void* dataPtr){
    float *xp, *yp, *de, *dn, *dv, *se, *sn, *sv;
    float stro1, dipo1, rakeo1, do1, nc1, ec1, a1, b1, x1, y1, p1, q1, ue1, un1, ux1, uy1, uz1, U11, U21, U31;
    float stro2, dipo2, rakeo2, do2, nc2, ec2, a2, b2, x2, y2, p2, q2, ue2, un2, ux2, uy2, uz2, U12, U22, U32;
    float stro3, dipo3, rakeo3, do3, nc3, ec3, a3, b3, x3, y3, p3, q3, ue3, un3, ux3, uy3, uz3, U13, U23, U33;
    float ux, uy, uz, dux, duy, duz;
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
    if (((x[ 5] / x[ 6]) > 1.0) && ((x[15] / x[16]) > 1.0) && ((x[25] / x[26]) > 1.0) && ((x[ 5] / x[ 6]) < 4.0) && ((x[15] / x[16]) < 4.0) && ((x[25] / x[26]) < 4.0)) {
        bp = 1;
        for (m = 0; m < stations; m++) {
            stro1  = x[ 3] * PI / 180.0;
            dipo1  = x[ 4] * PI / 180.0;
            rakeo1 = x[ 7] * PI / 180.0;
            U11    = cosf(rakeo1) * x[ 8];
            U21    = sinf(rakeo1) * x[ 8];
            U31    = x[ 9];

            stro2  = x[13] * PI / 180.0;
            dipo2  = x[14] * PI / 180.0;
            rakeo2 = x[17] * PI / 180.0;
            U12    = cosf(rakeo2) * x[18];
            U22    = sinf(rakeo2) * x[18];
            U32    = x[19];

            stro3  = x[23] * PI / 180.0;
            dipo3  = x[24] * PI / 180.0;
            rakeo3 = x[27] * PI / 180.0;
            U13    = cosf(rakeo3) * x[28];
            U23    = sinf(rakeo3) * x[28];
            U33    = x[29];

            /*
            * Converts fault coordinates (E,N,DEPTH) relative to centroid
            * into Okada's reference system (X,Y,D)
            */
            do1 = x[ 2] + sinf(dipo1) * x[ 6];
            ec1 = xp[m] - x[ 0] + cosf(stro1) * cosf(dipo1) * x[ 6] / 2.0;
            nc1 = yp[m] - x[ 1] - sinf(stro1) * cosf(dipo1) * x[ 6] / 2.0;
            x1  = cosf(stro1) * nc1 + sinf(stro1) * ec1 + x[ 5] / 2.0;
            y1  = sinf(stro1) * nc1 - cosf(stro1) * ec1 + cosf(dipo1) * x[ 6];

            do2 = x[12] + sinf(dipo2) * x[16];
            ec2 = xp[m] - x[10] + cosf(stro2) * cosf(dipo2) * x[16] / 2.0;
            nc2 = yp[m] - x[11] - sinf(stro2) * cosf(dipo2) * x[16] / 2.0;
            x2  = cosf(stro2) * nc2 + sinf(stro2) * ec2 + x[15] /2.0;
            y2  = sinf(stro2) * nc2 - cosf(stro2) * ec2 + cosf(dipo2) * x[16];

            do3 = x[22] + sinf(dipo3) * x[26];
            ec3 = xp[m] - x[20] + cosf(stro3) * cosf(dipo3) * x[26] / 2.0;
            nc3 = yp[m] - x[21] - sinf(stro3) * cosf(dipo3) * x[26] / 2.0;
            x3  = cosf(stro3) * nc3 + sinf(stro3) * ec3 + x[25] /2.0;
            y3  = sinf(stro3) * nc3 - cosf(stro3) * ec3 + cosf(dipo3) * x[26];

            /*
            * Variable substitution (independent from xi and eta)
            */
            p1 = y1 * cosf(dipo1) + do1 * sinf(dipo1);
            q1 = y1 * sinf(dipo1) - do1 * cosf(dipo1);

            p2 = y2 * cosf(dipo2) + do2 * sinf(dipo2);
            q2 = y2 * sinf(dipo2) - do2 * cosf(dipo2);

            p3 = y3 * cosf(dipo3) + do3 * sinf(dipo3);
            q3 = y3 * sinf(dipo3) - do3 * cosf(dipo3);

            /*
            * displacements
            */
            a1 = p1 - x[ 6];
            b1 = x1 - x[ 5];

            a2 = p2 - x[16];
            b2 = x2 - x[15];

            a3 = p3 - x[26];
            b3 = x3 - x[25];

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

            ue2 = -U12/(2.0*PI)*(ux_ss(x2,p2,q2,dipo2,NU)-ux_ss(x2,a2,q2,dipo2,NU)-ux_ss(b2,p2,q2,dipo2,NU)+ux_ss(b2,a2,q2,dipo2,NU)) -
            U22/(2.0*PI)*(ux_ds(x2,p2,q2,dipo2,NU)-ux_ds(x2,a2,q2,dipo2,NU)-ux_ds(b2,p2,q2,dipo2,NU)+ux_ds(b2,a2,q2,dipo2,NU)) +
            U32/(2.0*PI)*(ux_tf(x2,p2,q2,dipo2,NU)-ux_tf(x2,a2,q2,dipo2,NU)-ux_tf(b2,p2,q2,dipo2,NU)+ux_tf(b2,a2,q2,dipo2,NU));

            un2 = -U12/(2.0*PI)*(uy_ss(x2,p2,q2,dipo2,NU)-uy_ss(x2,a2,q2,dipo2,NU)-uy_ss(b2,p2,q2,dipo2,NU)+uy_ss(b2,a2,q2,dipo2,NU)) -
            U22/(2.0*PI)*(uy_ds(x2,p2,q2,dipo2,NU)-uy_ds(x2,a2,q2,dipo2,NU)-uy_ds(b2,p2,q2,dipo2,NU)+uy_ds(b2,a2,q2,dipo2,NU)) +
            U32/(2.0*PI)*(uy_tf(x2,p2,q2,dipo2,NU)-uy_tf(x2,a2,q2,dipo2,NU)-uy_tf(b2,p2,q2,dipo2,NU)+uy_tf(b2,a2,q2,dipo2,NU));

            ux2 = sinf(stro2) * ue2 - cosf(stro2) * un2;
            uy2 = cosf(stro2) * ue2 + sinf(stro2) * un2;

            ue3 = -U13/(2.0*PI)*(ux_ss(x3,p3,q3,dipo3,NU)-ux_ss(x3,a3,q3,dipo3,NU)-ux_ss(b3,p3,q3,dipo3,NU)+ux_ss(b3,a3,q3,dipo3,NU)) -
            U23/(2.0*PI)*(ux_ds(x3,p3,q3,dipo3,NU)-ux_ds(x3,a3,q3,dipo3,NU)-ux_ds(b3,p3,q3,dipo3,NU)+ux_ds(b3,a3,q3,dipo3,NU)) +
            U33/(2.0*PI)*(ux_tf(x3,p3,q3,dipo3,NU)-ux_tf(x3,a3,q3,dipo3,NU)-ux_tf(b3,p3,q3,dipo3,NU)+ux_tf(b3,a3,q3,dipo3,NU));

            un3 = -U13/(2.0*PI)*(uy_ss(x3,p3,q3,dipo3,NU)-uy_ss(x3,a3,q3,dipo3,NU)-uy_ss(b3,p3,q3,dipo3,NU)+uy_ss(b3,a3,q3,dipo3,NU)) -
            U23/(2.0*PI)*(uy_ds(x3,p3,q3,dipo3,NU)-uy_ds(x3,a3,q3,dipo3,NU)-uy_ds(b3,p3,q3,dipo3,NU)+uy_ds(b3,a3,q3,dipo3,NU)) +
            U33/(2.0*PI)*(uy_tf(x3,p3,q3,dipo3,NU)-uy_tf(x3,a3,q3,dipo3,NU)-uy_tf(b3,p3,q3,dipo3,NU)+uy_tf(b3,a3,q3,dipo3,NU));

            ux3 = sinf(stro3) * ue3 - cosf(stro3) * un3;
            uy3 = cosf(stro3) * ue3 + sinf(stro3) * un3;

            uz1 = -U11/(2.0*PI)*(uz_ss(x1,p1,q1,dipo1,NU)-uz_ss(x1,a1,q1,dipo1,NU)-uz_ss(b1,p1,q1,dipo1,NU)+uz_ss(b1,a1,q1,dipo1,NU)) -
            U21/(2.0*PI)*(uz_ds(x1,p1,q1,dipo1,NU)-uz_ds(x1,a1,q1,dipo1,NU)-uz_ds(b1,p1,q1,dipo1,NU)+uz_ds(b1,a1,q1,dipo1,NU)) +
            U31/(2.0*PI)*(uz_tf(x1,p1,q1,dipo1,NU)-uz_tf(x1,a1,q1,dipo1,NU)-uz_tf(b1,p1,q1,dipo1,NU)+uz_tf(b1,a1,q1,dipo1,NU));

            uz2 = -U12/(2.0*PI)*(uz_ss(x2,p2,q2,dipo2,NU)-uz_ss(x2,a2,q2,dipo2,NU)-uz_ss(b2,p2,q2,dipo2,NU)+uz_ss(b2,a2,q2,dipo2,NU)) -
            U22/(2.0*PI)*(uz_ds(x2,p2,q2,dipo2,NU)-uz_ds(x2,a2,q2,dipo2,NU)-uz_ds(b2,p2,q2,dipo2,NU)+uz_ds(b2,a2,q2,dipo2,NU)) +
            U32/(2.0*PI)*(uz_tf(x2,p2,q2,dipo2,NU)-uz_tf(x2,a2,q2,dipo2,NU)-uz_tf(b2,p2,q2,dipo2,NU)+uz_tf(b2,a2,q2,dipo2,NU));

            uz3 = -U13/(2.0*PI)*(uz_ss(x3,p3,q3,dipo3,NU)-uz_ss(x3,a3,q3,dipo3,NU)-uz_ss(b3,p3,q3,dipo3,NU)+uz_ss(b3,a3,q3,dipo3,NU)) -
            U23/(2.0*PI)*(uz_ds(x3,p3,q3,dipo3,NU)-uz_ds(x3,a3,q3,dipo3,NU)-uz_ds(b3,p3,q3,dipo3,NU)+uz_ds(b3,a3,q3,dipo3,NU)) +
            U33/(2.0*PI)*(uz_tf(x3,p3,q3,dipo3,NU)-uz_tf(x3,a3,q3,dipo3,NU)-uz_tf(b3,p3,q3,dipo3,NU)+uz_tf(b3,a3,q3,dipo3,NU));

            ux = ux1 + ux2 + ux3;
            uy = uy1 + uy2 + uy3;
            uz = uz1 + uz2 + uz3;

            dux = fabsf(ux - de[m]);
            duy = fabsf(uy - dn[m]);
            duz = fabsf(uz - dv[m]);

            if ((dux > se[m]) || (duy > sn[m]) || (duz > sv[m])) {
                return 0;
            }
        }
    }

    return bp;
}

}
