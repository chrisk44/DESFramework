#pragma once
#include "model.cu"

struct Limit {
	float lowerLimit;
	float upperLimit;
	unsigned long N;
};

struct ParallelFrameworkParameters {
	unsigned int D;
	unsigned int computeBatchSize;
	unsigned int batchSize;
	// ...
};

class ParallelFramework {
private:
	// Parameters
	Limit* limits;			// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters;
	Model* model;

	// Runtime variables
	long* idxSteps;			// Index steps for each dimension
	float* steps;			// Step for each dimension
	bool* results;			// An array of N0 * N1 * ... * ND
	

public:
	~ParallelFramework();
	int init(Limit* limits, ParallelFrameworkParameters& parameters, Model& model);
	int run();
	bool* getResults();
	bool getResultAt(float* point);

private:
	void scanDimension(int d, float* prevDims, bool* results, int startIdx);
};
