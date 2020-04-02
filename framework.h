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
	Limit* limits = NULL;			// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters = NULL;
	Model* model = NULL;

	// Runtime variables
	unsigned long* idxSteps = NULL;			// Index steps for each dimension
	float* steps = NULL;			// Step for each dimension
	bool* results = NULL;			// An array of N0 * N1 * ... * ND
	bool valid = false;
	unsigned long* toSendVector = NULL;		// An array of D elements, where every entry shows the next element of that dimension to be dispatched
	unsigned long* toSendVectorTmp = NULL;

public:
	ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters, Model& model);
	~ParallelFramework();

	int run();
	bool* getResults();
	bool getResultAt(float* point);
	bool isValid();

private:
	int masterThread();
	int slaveThread(int type);

	void scanDimension(int d, float* prevDims, bool* results, int startIdx);
	void getDataChunk(long* toCalculate, int *numOfElements);
};
