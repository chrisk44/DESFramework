#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace mogi {

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
 * azimuth function
 */
__host__ __device__ __inline__ float t(float x1, float y1, float x2, float y2)
{
    float	a, dx, dy;

    dx = x2 - x1;
    dy = y2 - y1;
    a  = atanf(fabsf(dx) / fabsf(dy));

    if ( dx > 0.0 && dy > 0.0) {
        return(a);
    } else if (dx > 0.0 && dy < 0.0) {
        return(PI - a);
    } else if (dx < 0.0 && dy < 0.0) {
        return(PI + a);
    } else {
        return(2.0 * PI - a);
    }
}

/******************************************************************************/

/*
 * horizontal displacement function
 */

__host__ __device__ __inline__ float f(float x, float y, float z, float dv, float xl, float yl, float zl)
{
    float	dx = x - xl, dy = y - yl, dz = z + zl;
    return((3.0 * dv * sqrtf(dx * dx + dy * dy)) / ((4.0 * PI) * powf(dz * dz + dx * dx + dy * dy, 1.5)));
}

/******************************************************************************/

/*
 * vertical displacement function
 */

__host__ __device__ __inline__ float h(float x, float y, float z, float dv, float xl, float yl, float zl)
{
    float	dx = x - xl, dy = y - yl, dz = z + zl;
    return((3.0 *  dv * dz) / ((4.0 * PI) * powf(dz * dz + dx * dx + dy * dy, 1.5)));
}

/******************************************************************************/

}
