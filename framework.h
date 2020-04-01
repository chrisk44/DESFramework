#pragma once
#include "model.cu"

struct Limit {
	float lowerLimit;
	float upperLimit;
	float step;
};

struct ParallelFrameworkParameters {
	int D;
	int computeBatchSize;
	int batchSize;
	// ...
};

class ParallelFramework {
private:
	Limit* limits;			// This must be an array of length = parameters.D
	long* N;				// This must be an array of length = parameters.D
	long* idxSteps;			// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters;
	Model* model;
	bool* results;
	

public:
	int init(Limit* limits, ParallelFrameworkParameters& parameters, Model& model);
	int run();
	bool* getResults();
	bool getResultAt(float* point);

private:
	void scanDimension(int d, float* prevDims, bool* results, int startIdx);
};
