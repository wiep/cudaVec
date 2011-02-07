#include "cudavec.h"
#include <cuda.h>
#include <cassert>

namespace cudaVector {

cudaVec& cudaVec::operator+=(const float& f){
	*this = *this + f;
	return *this;
}

cudaVec& cudaVec::operator-=(const float& f){
	*this = *this - f;
	return *this;
}

cudaVec& cudaVec::operator*=(const float& f){
	*this = *this * f;
	return *this;
}

cudaVec& cudaVec::operator/=(const float& f){
	*this = *this / f;
	return *this;
}

cudaVec cudaVec::operator-(){
	return *this * -1;
}


std::ostream& operator<<(std::ostream& os, const cudaVec& vec) {
	if (vec.size() > 0) {
		os <<"[";
		for (unsigned int i = 0; i < vec.size()-1; ++i) {
			os << vec[i] << ", ";
		}
		os << vec[vec.size() - 1] << "]";
	}else{
		os << "[]";
	}
	return os;
}

int getNextKernelID() {
	static int kernel_couter = 0;
	return kernel_couter++;
}

#ifdef REUSE_KERNELS
bool file_exists(std::string filename) {
	struct stat stFileInfo;
	int result = stat(filename.c_str(), &stFileInfo);
	return result == 0;
}
#endif

CUdevice cuDevice;
CUcontext cuContext;
int maxThreadsPerBlock;
int maxBlocksPerGrid;

int initCUDA() {
	CUresult result;
	static bool cuda_driver_init = false;
	if (!cuda_driver_init) {

		//Init CUDA Driver
		result = cuInit(0);
		assert(result == CUDA_SUCCESS);

		//Get number of cudasupporting devices
		int deviceCount = 0;
		result = cuDeviceGetCount(&deviceCount);
		assert(result == CUDA_SUCCESS);

		if (deviceCount == 0) {
			std::cerr << "There is no device supporting CUDA." << std::endl;
			exit(1);
		}

		//select device
		result = cuDeviceGet(&cuDevice, 0);
		assert(result == CUDA_SUCCESS);

		//get device name
		char deviceName[256];
		cuDeviceGetName(deviceName, 256, cuDevice);

		//get some attributes
		int blockDim[3];
		int gridDim[3];
		cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuDevice);
		cuDeviceGetAttribute( &blockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, cuDevice);
		cuDeviceGetAttribute( &blockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, cuDevice);
		cuDeviceGetAttribute( &blockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, cuDevice);
		cuDeviceGetAttribute( &gridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, cuDevice);
		cuDeviceGetAttribute( &gridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, cuDevice);
		cuDeviceGetAttribute( &gridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, cuDevice);

		maxThreadsPerBlock = maxThreadsPerBlock < blockDim[0] ? maxThreadsPerBlock : blockDim[0];
		maxBlocksPerGrid = gridDim[0];

		std::cerr << "-----------------------------------------------------------------------" << std::endl;
		std::cerr << "Using Device0: " << deviceName << std::endl;
		std::cerr << "   Maximum threads per block:      " << maxThreadsPerBlock << std::endl;
		std::cerr << "   Maximum block dimension:        " << blockDim[0] << " x "  << blockDim[1] << " x " 
			<< blockDim[2] << std::endl;
		std::cerr << "   Maximum grid dimension:         " << gridDim[0] << " x "  << gridDim[1] << " x " 
			<< gridDim[2] << std::endl;

		std::cerr << "   The vector size is now limited by the amount of device memory\n   or the maximum value of an unsigned int.\n   Further testing needed to substantiate this.\n";
		std::cerr << "-----------------------------------------------------------------------" << std::endl;

		result = cuCtxCreate(&cuContext, 0, cuDevice);
		assert(result == CUDA_SUCCESS);


		cuda_driver_init = true;
	}

	static bool cuBlas_init = false;
	if (!cuBlas_init) {
		cublasInit();
		//TODO test result.
		cuBlas_init = true;
	}
	return 0;
}

int shutdownCUDA() {
	//TODO destroy context
	//TODO test cuBlas_init && cuda_driver_init 
	cublasShutdown();
	return 0;
}

} // namespace cudaVector
