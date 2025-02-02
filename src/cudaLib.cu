
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		y[idx] = scale * x[idx] + y[idx];
	}
}

int runGpuSaxpy(int vectorSize) {

	// std::cout << "Hello GPU Saxpy!\n";
	std::cout << "Vector Size: " << vectorSize << " \n";
	int size = vectorSize * sizeof(float);

	float* x = (float*) malloc(size);
	float* y = (float*) malloc(size);
	float* reference_y = (float*) malloc(size);
	// float a = 3.2f;
	float a = (float) (rand() % 1000);

	// Generate vectors
	for (int i = 0; i < vectorSize; i++) {
		// x[i] = 1.0f;
		// y[i] = 2.0f;
		x[i] = (float) (rand() % 1000);
		y[i] = (float) (rand() % 1000);


		// A*X + Y should return all 5's
		// reference_y[i] = a * x[i] + y[i];
	}
	for (int i = 0; i < vectorSize; i++) {
		reference_y[i] = a * x[i] + y[i];
	}

	float* gpu_x;
	float* gpu_y;

	// Transfer from x, y to gpu_x, gpu_y
	cudaMalloc(&gpu_x, size);
	cudaMemcpy(gpu_x, x, size, cudaMemcpyHostToDevice);
	cudaMalloc(&gpu_y, size);
	cudaMemcpy(gpu_y, y, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
	saxpy_gpu<<<blocksPerGrid, threadsPerBlock>>>(gpu_x, gpu_y, a, vectorSize);


	// Transfer back from gpu_x, gpu_y to x, y
	// cudaMemcpy(x, gpu_x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, gpu_y, size, cudaMemcpyDeviceToHost);

	// Free device memory
	// cudaFree(gpu_x);
	cudaFree(gpu_y);

	// Check if computations were correct
	int errorCount = 0;
	float error = 0.001;
	for (int i = 0; i < vectorSize; i++) {
		if ((y[i] < (reference_y[i] - error)) || (y[i] > (reference_y[i] + error))) {
			errorCount++;
			std::cout << "MISMATCH: GOT " << y[i] << " EXPECTED " << reference_y[i] << " \n";
		}
	}

	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";
	free(x); free(y); free(reference_y);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_id >= pSumSize) {
		return;
	}

	curandState_t rng;
	curand_init(clock64(), thread_id, 0, &rng);

	uint64_t hitcnt = 0;

	for (uint64_t i = 0; i < sampleSize; i++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);

		if ((x*x + y*y) <= 1.0f) {
			hitcnt++;
		}
	}

	pSums[thread_id] = hitcnt;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	// Generate arrays in both GPU and CPU
	uint64_t* hits = (uint64_t *) malloc(sizeof(uint64_t) * generateThreadCount);
	uint64_t* gpu_hits;
	cudaMalloc(&gpu_hits, generateThreadCount * sizeof(uint64_t));

	// Kernel call
	int threadsPerBlock = 256;
	int blocksPerGrid = (generateThreadCount + threadsPerBlock - 1) / threadsPerBlock;
	generatePoints<<<blocksPerGrid, threadsPerBlock>>>(gpu_hits, generateThreadCount, sampleSize);

	cudaMemcpy(hits, gpu_hits, generateThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	cudaFree(gpu_hits);

	uint64_t total_hits = 0;
	for (int i = 0; i < generateThreadCount; i++) {
		total_hits += hits[i];
	}
	free(hits);

	printf("Total hits: %d\n", total_hits);
	printf("Total hits as a double: %lf\n", (double) total_hits);
	approxPi = (double) total_hits / (double) (generateThreadCount * sampleSize) * 4.0;


	return approxPi;
}
