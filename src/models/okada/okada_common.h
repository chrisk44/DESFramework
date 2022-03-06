#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace okada {

#define EPS	(1.0E-7)
/*
 * Isotropic Poisson's ratio
 */
#define NU	(0.25)

#ifndef TOPINV_PI
#define TOPINV_PI
#if defined(__CUDACC__)
__device__ const float PI = (float)M_PI;
#else
const float PI = (float)M_PI;
#endif
#endif

/******************************************************************************/

/*
 * I[1-5] displacement subfunctions [equations (28) (29) p. 1144-1145]
 */

__host__ __device__ __inline__ float I5(float xi, float eta, float q, float dip, float nu, float R, float db)
{
    float	X = sqrtf(xi * xi + q * q);

    if (cosf(dip) > EPS) {
        return((1.0 - 2.0 * nu) * 2.0 / cosf(dip) * atanf((eta * (X + q * cosf(dip)) + X * (R + X) * sinf(dip)) / (xi * (R + X) * cosf(dip))));
    } else {
        return(-(1.0 - 2.0 * nu) * xi * sinf(dip) / (R + db));
    }
}

__host__ __device__ __inline__ float I4(float db, float eta, float q, float dip, float nu, float R)
{
    if (cosf(dip) > EPS) {
        return((1.0 - 2.0 * nu) * 1.0 / cosf(dip) * (logf(R + db) - sinf(dip) * logf(R + eta)));
    } else {
            return(-(1.0 - 2.0 * nu) * q / (R + db));
    }
}

__host__ __device__ __inline__ float I3(float eta, float q, float dip, float nu, float R)
{
    float	yb = eta * cosf(dip) + q * sinf(dip);
    float	db = eta * sinf(dip) - q * cosf(dip);

    if (cosf(dip) > EPS) {
        return((1.0 - 2.0 * nu) * (yb / (cosf(dip) * (R + db)) - logf(R + eta)) + sinf(dip) / cosf(dip) * I4(db, eta, q, dip, nu, R));
    } else {
        return((1.0 - 2.0 * nu) / 2.0 * (eta / (R + db) + yb * q / ((R + db) * (R + db)) - logf(R + eta)));
    }
}

__host__ __device__ __inline__ float I2(float eta, float q, float dip, float nu, float R)
{
    return((1.0 - 2.0 * nu) * (-logf(R + eta)) - I3(eta, q, dip, nu, R));
}

__host__ __device__ __inline__ float I1(float xi, float eta, float q, float dip, float nu, float R)
{
    float	db = eta * sinf(dip) - q * cosf(dip);

    if (cosf(dip) > EPS) {
        return((1.0 - 2.0 * nu) * (-xi / (cosf(dip) * (R + db))) - sinf(dip) / cosf(dip) * I5(xi, eta, q, dip, nu, R, db));
    } else {
        return(-(1.0 - 2.0 * nu) / 2.0 * xi * q / ((R + db) * (R + db)));
    }
}

/*
 * strike-slip displacement subfunctions [equation (25) p. 1144]
 */

__host__ __device__ __inline__ float ux_ss(float xi, float eta, float q, float dip, float nu)
{
    float	R = sqrtf(xi * xi + eta * eta + q * q);

    return(xi * q / (R * (R + eta)) + atanf(xi * eta / (q * R)) + I1(xi, eta, q, dip, nu, R) * sinf(dip));
}

__host__ __device__ __inline__ float uy_ss(float xi, float eta, float q, float dip, float nu)
{
    float	R = sqrtf(xi * xi + eta * eta + q * q);

    return((eta * cosf(dip) + q * sinf(dip)) * q / (R * (R + eta)) + q * cosf(dip) / (R + eta) + I2(eta, q, dip, nu, R) * sinf(dip));
}

__host__ __device__ __inline__ float uz_ss(float xi, float eta, float q, float dip, float nu)
{
    float	R  = sqrtf(xi * xi + eta * eta + q * q);
    float	db = eta * sinf(dip) - q * cosf(dip);

    return((eta * sinf(dip) - q * cosf(dip)) * q / (R * (R + eta)) + q * sinf(dip) / (R + eta) + I4(db, eta, q, dip, nu, R) * sinf(dip));
}

/*
 * dip-slip displacement subfunctions [equation (26) p. 1144]
 */
__host__ __device__ __inline__ float ux_ds(float xi, float eta, float q, float dip, float nu)
{
    float	R = sqrtf(xi * xi + eta * eta + q * q);
    return(q / R - I3(eta, q, dip, nu, R) * sinf(dip) * cosf(dip));
}

__host__ __device__ __inline__ float uy_ds(float xi, float eta, float q, float dip, float nu)
{
    float	R = sqrtf(xi * xi + eta * eta + q * q);

    return((eta * cosf(dip) + q * sinf(dip)) * q / (R * (R + xi)) + cosf(dip) * atanf(xi * eta / (q * R)) - I1(xi, eta, q, dip, nu, R) * sinf(dip) * cosf(dip));
}

__host__ __device__ __inline__ float uz_ds(float xi, float eta, float q, float dip, float nu)
{
    float	R  = sqrtf(xi * xi + eta * eta + q * q);
    float	db = eta * sinf(dip) - q * cosf(dip);

    return(db * q / (R * (R + xi)) + sinf(dip) * atanf(xi * eta / (q * R)) - I5(xi, eta, q, dip, nu, R, db) * sinf(dip) * cosf(dip));
}

/*
 * tensile fault displacement subfunctions [equation (27) p. 1144]
 */
__host__ __device__ __inline__ float ux_tf(float xi, float eta, float q, float dip, float nu)
{
    float	R = sqrtf(xi * xi + eta * eta + q * q);

    return((q * q) / (R * (R + eta)) - I3(eta, q, dip, nu, R) * sinf(dip) * sinf(dip));
}

__host__ __device__ __inline__ float uy_tf(float xi, float eta, float q, float dip, float nu)
{
    float	R = sqrtf(xi * xi + eta * eta + q * q);

    return(-(eta * sinf(dip) - q * cosf(dip)) * q / (R * (R + xi)) - sinf(dip) * (xi * q / (R * (R + eta)) - atanf(xi * eta / (q * R))) - I1(xi, eta, q, dip, nu, R) * sinf(dip) * sinf(dip));
}

__host__ __device__ __inline__ float uz_tf(float xi, float eta, float q, float dip, float nu)
{
    float	R  = sqrtf(xi * xi + eta * eta + q * q);
    float	db = eta * sinf(dip) - q * cosf(dip);

    return((eta * cosf(dip) + q * sinf(dip)) * q / (R * (R + xi)) + cosf(dip) * (xi * q / (R * (R + eta)) - atanf(xi * eta / (q * R))) - I5(xi, eta, q, dip, nu, R, db) * sinf(dip) * sinf(dip));
}

/******************************************************************************/

}
