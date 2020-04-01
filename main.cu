
#include "cuda_runtime.h"

#include <stdio.h>
#include "model.cu"

class MyModel : Model{
public:
    __host__ bool validate_cpu(float *point){
        return true;
    }

    __device__ bool validate_gpu(float *point){
        return true;
    }

    bool toBool(){ return true; }
};

__global__ void validate_kernel(Model *model, float *points, bool *results){
    int i = threadIdx.x;
    
    float point = i;

    bool result = model->validate_gpu(&point);
    results[i] = result;
}

int main(){
    

    return 0;
}
