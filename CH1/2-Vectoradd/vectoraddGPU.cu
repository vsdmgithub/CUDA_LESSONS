// HEADER FILES
#include <cuda_runtime.h>
#include <stdio.h>

// ERROR HANDING MACRO
#define CHECK(call)                                                                     \
{											\
	const cudaError_t_error = call;							\
	if ( error != cudaSuccess )							\
	{										\
		printf("Error: %s:%d, ", __FILE__, __LINE__ );				\
		printf("Code: %d,reason: %s \n ",error, cudaGetErrorString( error ));	\
		exit(1);								\
	}										\
}

// VERIFICATION OF KERNEL 
void checkResult( float *hostRef, float *gpuRef, const int N ) 
{
	double epsilon		=	1.0E-8;
	bool match		=	1;
	for( int i = 0; i < N; i++ )
	{
		if ( abs( hostRef[i] - gpuRef[i] > epsilon ) )
		{
			match	=	0;
			printf(" Arrays do not match ! \n ");
			printf(" host %5.2f gpu %5.2f at current %d \n ", hostRef[i],gpuRef[i],i);	
			break;
		}
	} 
	if( match ) printf("Arrays do match ! \n \n ");
}

// INITIALISING DATA 
void initialData( float *ip, int size ) 
{
	time_t t;
	srand( (unsigned) time(&t) );
	// GENERATE DIFFERENT SEED FOR RANDOM NUMBER
	
	for( int i = 0; i < size; i++ )
	{
		ip[i]	= 	(float) ( rand() & 0XFF ) / 10.0f;
	}
}

void sumArraysOnHost( float *A, float *B, float *C, const int N )
{
	for( int idx = 0; idx < N; idx++ )
	C[idx]	=	A[idx] + B[idx];
}

__global__ void sumArraysOnGPU( float *A, float *B, float *C) 
{
	int i	=	blockIdx.x * blockDim.x + threadIdx.x;
	C[i]	=	A[i] + B[i];
	printf("threadIdx: (%d,%d,%d) | blockIdx:(%d,%d,%d)| blockDim: (%d,%d,%d)| gridDim:(%d,%d,%d)|    \n Array location : %d \n",threadIdx.x,threadIdx.y,threadIdx.z, blockIdx.x,blockIdx.y,blockIdx.z, blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z,i);
}

int main( int argc, char **argv )
{
	printf(" %s Starting... \n", argv[0]);

	// SET UP DEVICE
	int dev		= 0;
	cudaSetDevice(dev);

	// SET UP DATA SIZE OF VECTORS
	int nElem	= 64;
	printf("Vector size %d \n ", nElem);

	// MALLOC HOST MEMORY
	size_t nBytes	= nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A		= (float *) malloc(nBytes);
	h_B		= (float *) malloc(nBytes);
	hostRef		= (float *) malloc(nBytes);
	gpuRef		= (float *) malloc(nBytes);

	// INITIALIZE DATA FROM HOST SIDE
	initialData( h_A, nElem );
	initialData( h_B, nElem );

	memset( hostRef, 0, nBytes);
	memset(  gpuRef, 0, nBytes);

	// MALLOC DEVICE GLOBAL MEMORY
	float *d_A, *d_B, *d_C;
	cudaMalloc( (float**)&d_A, nBytes);
	cudaMalloc( (float**)&d_B, nBytes);
	cudaMalloc( (float**)&d_C, nBytes);

	// TRANSFER DATA FROM HOST TO DEVICE
	cudaMemcpy( d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	// INVOKE KERNEL AT HOST SIDE
	// dim3 block ( nElem );
	dim3 block ( 16 );
	dim3 grid  ( nElem / block.x );

	sumArraysOnGPU<<< grid, block >>>( d_A, d_B, d_C );
	printf("Execution configuration <<< %d, %d >>> \n ",grid.x,block.x);

	// COPY KERNEL RESULT BACK TO HOST SIDE
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	// ADD VECTOR FROM HOST SIDE FOR CHECKING 
	sumArraysOnHost( h_A, h_B, hostRef, nElem);

	// CHECK DEVICE RESULTS
	checkResult( hostRef, gpuRef, nElem);

	// FREE DEVICE GLOBAL MEMORY
	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

	// FREE HOST MEMORY
	free( h_A );
	free( h_B );
	free( hostRef );
	free( gpuRef );
	
	return(0);
}

