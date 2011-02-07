// vim:foldmethod=marker
// vim:foldmarker={{{,}}}
#ifndef _CUDA_VECTOR_H_
#define _CUDA_VECTOR_H_

// TODO cudaMatrix, template specialization for CUBLAS functions like y = A * x + y and the like
// TODO optimize generated kernels. 



#ifndef CUDA_KERNEL_TMP_DIR 
#define CUDA_KERNEL_TMP_DIR "/tmp"
#endif

#ifndef NVCC
#define NVCC "/opt/cuda/bin/nvcc"
#endif

#ifndef NVCC_FLAGS
#define NVCC_FLAGS "--compiler-bindir=/usr/bin/g++-4.3"
#endif



#ifndef ALIGN_UP
#define ALIGN_UP(offset, alignment) \
	(offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)
#endif

#include <sys/stat.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <sstream>
#include <cuda.h>
#include <cublas.h>

#include <thrust/device_vector.h>

namespace cudaVector {

int getNextKernelID();
int initCUDA();
int shutdownCUDA();
extern CUdevice cuDevice;
extern CUcontext cuContext;
extern int maxThreadsPerBlock;
extern int maxBlocksPerGrid;

template<class E>
class AssignmentExpression;

template <class E>
class Expression{
public:
	operator const E&() const { return static_cast<const E&> (*this); }
};


// {{{ cudaVec
class cudaVec :  public thrust::device_vector<float>, public Expression<cudaVec>
{
public:
	//typedefs
	typedef thrust::device_vector<float> Parent;
	typedef Parent::size_type size_type;

	//constructor
	cudaVec(void) : Parent() {}
	cudaVec(size_type size) : Parent(size, 0) {}
	cudaVec(size_type size, float value) : Parent(size, value) {}
	cudaVec(const thrust::device_vector<float> &vec) : Parent(vec) {}
	cudaVec(const cudaVec &vec) : Parent(vec) {}

	float* rawDataPointer(void) { return this->data().get(); }

	template<class E>
	cudaVec(Expression<E> const &e) : Parent(static_cast<const E&>(e).size()) {
		//eval expression.
		AssignmentExpression<E> ass(*this, e);
		std::cerr << "evaluate kernel" << ass.kernelID << std::endl;
		ass.eval();
	}

	template<class E>
	cudaVec& operator=(Expression<E> const &r) {
		const E& _r = r;
		this->resize(_r.size());
		// TODO maybe this would also work with a static variable instead of AssignmentExpression
		AssignmentExpression<E> ass(*this, r);
		std::cerr << "evaluate kernel" << ass.kernelID << std::endl;
		ass.eval();
		return *this;
	}

	static void fillCudaStrings(char& current_var, std::stringstream& eval_line, std::stringstream& def_line) { 
		def_line << "float* " << current_var;
		eval_line << current_var++ << "[idx]";
	}

	static void fillName(std::stringstream& ss) {
		ss << "_" << "cudaVec" << "_0";
	}

	void setCudaParam(CUfunction& cuFunction, int& offset) const { 
		CUresult result;
		void *ptr = (void*) this->data().get();
		ALIGN_UP(offset, __alignof(ptr));
		result = cuParamSetv(cuFunction, offset, &ptr, sizeof(ptr));
		assert(result == CUDA_SUCCESS);
		offset += sizeof(ptr);
	}

	cudaVec& operator+=(const float& f);
	template<class E> cudaVec& operator+=(const Expression<E>& e){
		*this = *this + e;
		return *this;
	}
	cudaVec& operator-=(const float& f);
	template<class E> cudaVec& operator-=(const Expression<E>& e){
		*this = *this - e;
		return *this;
	}

	cudaVec& operator*=(const float& f);
	template<class E> cudaVec& operator*=(const Expression<E>& e){
		*this = *this * e;
		return *this;
	}

	cudaVec& operator/=(const float& f);
	template<class E> cudaVec& operator/=(const Expression<E>& e){
		*this = *this / e;
		return *this;
	}

	cudaVec& operator++(){
		*this += 1;
		return *this;
	}

	cudaVec operator++(int){
		cudaVec tmp(*this);
		*this += 1;
		return tmp;
	}

	cudaVec& operator--(){
		*this -= 1;
		return *this;
	}

	cudaVec operator--(int){
		cudaVec tmp(*this);
		*this -= 1;
		return tmp;
	}

	cudaVec operator-();

	friend std::ostream& operator<<(std::ostream& os, const cudaVec& vec);

};
//}}}

// {{{ AssigmentExpression
template<class E>
class AssignmentExpression {
	public:
		AssignmentExpression(cudaVec &vec, Expression<E> const& r)
			: _vec(vec), _r(r) {
				 assert (_vec.size() == _r.size());
		}


		static char last_var;
		static int kernelID;
		static CUmodule cuModule;
		static CUfunction cuFunction;
		static CUfunction cuFunction_small_vec;
		void eval();
	private:
		cudaVec& _vec;
		E const& _r;
		static std::string getCudaString();
		static std::string getName();
		static int init();
		

};

template<class E>
char AssignmentExpression<E>::last_var = 'a';

template<class E>
CUmodule AssignmentExpression<E>::cuModule;

template<class E>
CUfunction AssignmentExpression<E>::cuFunction;
template<class E>
CUfunction AssignmentExpression<E>::cuFunction_small_vec;

template<class E>
int AssignmentExpression<E>::kernelID = 
	AssignmentExpression<E>::init();

template<class E>
void AssignmentExpression<E>::eval() {

	//calculate thread layout.
	unsigned int size = _vec.size();
	unsigned int threadsPerBlock = maxThreadsPerBlock;
	unsigned int blocksPerGrid = 
			(size + threadsPerBlock - 1) / threadsPerBlock; // ceil(size/threadsPerBlock)
	if (blocksPerGrid > maxBlocksPerGrid)
		blocksPerGrid = maxBlocksPerGrid;
	//threadsPerBlock = 1;
	//blocksPerGrid = 1;
	unsigned int threads = blocksPerGrid * threadsPerBlock;

	if (size <= threads) {
		//use small kernel

		//set expression parameters
		CUresult result;
		int offset = 0;
		_vec.setCudaParam(cuFunction_small_vec, offset);
		_r.setCudaParam(cuFunction_small_vec, offset);

		//set size parameter
		ALIGN_UP(offset, __alignof(size));
		result = cuParamSeti(cuFunction_small_vec, offset, size);
		assert(result == CUDA_SUCCESS);
		offset += sizeof(size);
		result = cuParamSetSize(cuFunction_small_vec, offset);
		assert(result == CUDA_SUCCESS);

		//launch kernel
		assert(blocksPerGrid <= maxBlocksPerGrid);
		result = cuFuncSetBlockShape(cuFunction_small_vec, threadsPerBlock, 1, 1);
		assert(result == CUDA_SUCCESS);
		result = cuLaunchGrid(cuFunction_small_vec, blocksPerGrid, 1);
		assert(result == CUDA_SUCCESS);
	}else{
		//use normal kernel
		
		//set expression parameters
		CUresult result;
		int offset = 0;
		_vec.setCudaParam(cuFunction, offset);
		_r.setCudaParam(cuFunction, offset);

		//set size parameter
		ALIGN_UP(offset, __alignof(size));
		result = cuParamSeti(cuFunction, offset, size);
		assert(result == CUDA_SUCCESS);
		offset += sizeof(size);
		result = cuParamSetSize(cuFunction, offset);
		assert(result == CUDA_SUCCESS);

		//set number of threads parameter
		ALIGN_UP(offset, __alignof(threads));
		result = cuParamSeti(cuFunction, offset, threads);
		assert(result == CUDA_SUCCESS);
		offset += sizeof(threads);
		result = cuParamSetSize(cuFunction, offset);
		assert(result == CUDA_SUCCESS);

		//launch kernel
		assert(blocksPerGrid <= maxBlocksPerGrid);
		result = cuFuncSetBlockShape(cuFunction, threadsPerBlock, 1, 1);
		assert(result == CUDA_SUCCESS);
		result = cuLaunchGrid(cuFunction, blocksPerGrid, 1);
		assert(result == CUDA_SUCCESS);
	}
}

template<class E>
std::string AssignmentExpression<E>::getName() {
	std::stringstream ss;

	E::fillName(ss);

	return ss.str();
}

template<class E>
std::string AssignmentExpression<E>::getCudaString() {
	char &current_var = last_var;
	std::stringstream eval_line, def_line;
	cudaVec::fillCudaStrings(current_var, eval_line, def_line);
	def_line << ", ";
	eval_line << " = ";
	E::fillCudaStrings(current_var, eval_line, def_line);
	eval_line << ";";

	std::stringstream ss;

	//kernel for arbitrary large vectors.
	ss << "extern \"C\" __global__ void kernel( ";
	ss << def_line.str() << ", unsigned int vector_size, unsigned int number_of_used_threads ) { \n";
	ss << "\tint idx = blockDim.x * blockIdx.x + threadIdx.x; \n";
	ss << "\tfor(unsigned int i = 0; i < ";
	ss << "(vector_size + number_of_used_threads - 1) / number_of_used_threads; ++i) {\n";
	ss << "\t\tif(idx < vector_size) { \n";
	ss << "\t\t\t" << eval_line.str() << "\n";
	ss << "\t\t\tidx += number_of_used_threads;\n";
	ss << "\t\t}\n";
	ss << "\t}\n";
	ss << "}\n\n\n\n";

	//kernel for small vectorsize. One thread per element.
	ss << "extern \"C\" __global__ void kernel_small_vec( ";
	ss << def_line.str() << ", unsigned int vector_size) { \n";
	ss << "\tint idx = blockDim.x * blockIdx.x + threadIdx.x; \n";
	ss << "\tif(idx < vector_size)\n";
	ss << "\t" << eval_line.str() << "\n";
	ss << "}";

	//alternative kernel code.
	/*ss << "extern \"C\" __global__ void kernel( ";
	ss << def_line.str() << ", unsigned int vector_size, unsigned int number_of_used_threads ) { \n";
	ss << "\tfor(unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; ";
	ss << "idx < vector_size; idx += number_of_used_threads) {\n";
	ss << "\t\t" << eval_line.str() << "\n";
	ss << "\t}\n";
	ss << "}";*/

	//std::cerr << ss.str() << std::endl;
	return ss.str();
}

template<class E>
int AssignmentExpression<E>::init() {
	initCUDA();
	CUresult result;
	std::string cudaFunctionString = AssignmentExpression<E>::getCudaString();
	int id = getNextKernelID();

	//generate kernel filenames
	std::stringstream ss;
	std::string kernel_filename, kernel_comp_filename;

#ifdef REUSE_KERNELS
	ss << CUDA_KERNEL_TMP_DIR << "/kernel_" << AssignmentExpression<E>::getName() << ".cu";
#else
	#ifdef NDEBUG
		//get processId
		pid_t pid = getpid();

		ss << CUDA_KERNEL_TMP_DIR << "/cvk" << pid << "_" << id << ".cu";
	#else
		ss << CUDA_KERNEL_TMP_DIR << "/cvk" << id << ".cu";
	#endif
#endif

	kernel_filename = ss.str();
	kernel_comp_filename = kernel_filename.substr(0, kernel_filename.size() - 2) + "ptx";
	// TODO if kernel_comp_filename exists and is not readable, or if either file is not writable, choose a different name

#ifdef REUSE_KERNELS
	// try to load compiled cuda kernel
	bool valid_kernel = (cuModuleLoad(&cuModule, kernel_comp_filename.c_str()) == CUDA_SUCCESS);
	// TODO errors in ptx file can cause segfault at next cuModuleLoad()
	valid_kernel &= (cuModuleGetFunction(&cuFunction, cuModule, "kernel") == CUDA_SUCCESS);
	valid_kernel &= (cuModuleGetFunction(&cuFunction_small_vec, cuModule, "kernel_small_vec") == CUDA_SUCCESS);
	if (!valid_kernel) {
#endif

#ifndef NDEBUG
	std::cout << "Compiling " << kernel_filename << "..." << std::endl;
#endif

	//save cudaFuctionString in file.
	std::ofstream kernel_file;
	kernel_file.open(kernel_filename.c_str());
	kernel_file << cudaFunctionString;
	kernel_file.close();
	
	//compile kernel to ptx
	int nvcc_exit_status = system(
		(std::string(NVCC) + " -ptx " + NVCC_FLAGS + " " + kernel_filename 
		 + " -o " + kernel_comp_filename).c_str()
	);
		
	if (nvcc_exit_status) {
		std::cerr << "ERROR: nvcc exits with status code: " << nvcc_exit_status << std::endl;
		exit(1);
	}

	//load compiled cuda kernel
	result = cuModuleLoad(&cuModule, kernel_comp_filename.c_str());
	assert(result == CUDA_SUCCESS);
	result =  cuModuleGetFunction(&cuFunction, cuModule, "kernel");
	assert(result == CUDA_SUCCESS);
	result =  cuModuleGetFunction(&cuFunction_small_vec, cuModule, "kernel_small_vec");
	assert(result == CUDA_SUCCESS);

#ifdef REUSE_KERNELS
	}
#endif
	
	return id;
}
//}}}

// {{{ make_binary_scalar_expression_infix_operator
#define make_binary_scalar_expression_infix_operator(CLASSNAME, OPSYMBOL, CUDAOPSYMBOL) \
template <typename E> \
class CLASSNAME : public Expression<CLASSNAME<E> > { \
	public: \
		float const& _l; \
		E const& _r; \
 \
		CLASSNAME( float const& l, Expression<E> const& r) \
			: _l(l), _r(r) { } \
 \
		unsigned int size() const { return _r.size(); } \
 \
		static void fillCudaStrings(char& current_var, std::stringstream& eval_line, \
				std::stringstream& def_line) {  \
			eval_line << "(" << current_var; \
			eval_line	<< " " << CUDAOPSYMBOL << " " ; \
			def_line << "float " << current_var++ << ", "; \
			E::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< ")"; \
		} \
 \
		static void fillName(std::stringstream& ss) { \
			ss << "_" << #CLASSNAME << "_1"; \
			E::fillName(ss); \
		} \
 \
		void setCudaParam(CUfunction& cuFunction, int& offset) const { \
			 \
			CUresult result; \
			ALIGN_UP(offset, __alignof(_l)); \
			result = cuParamSetf(cuFunction, offset, _l); \
			assert(result == CUDA_SUCCESS); \
			offset += sizeof(_l); \
 \
			_r.setCudaParam(cuFunction, offset); \
		}; \
 \
}; \
 \
template<class E> \
CLASSNAME<E> operator OPSYMBOL( float const &l, Expression<E> const &r) { \
	return CLASSNAME<E> (l, r); \
}
//z}}}

// {{{ make_binary_expression_scalar_infix_operator
#define make_binary_expression_scalar_infix_operator(CLASSNAME, OPSYMBOL, CUDAOPSYMBOL) \
template <typename E> \
class CLASSNAME : public Expression<CLASSNAME<E> > { \
	public: \
		E const& _l; \
		float const& _r; \
 \
		CLASSNAME( Expression<E>const& l, float  const& r) \
			: _l(l), _r(r) { } \
 \
		unsigned int size() const { return _l.size(); } \
 \
		static void fillCudaStrings(char& current_var, std::stringstream& eval_line, \
				std::stringstream& def_line) {  \
			eval_line << "("; \
			E::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< " " << CUDAOPSYMBOL << " " ; \
			eval_line	<< current_var << ")"; \
			def_line << ", " << "float " << current_var++; \
		} \
 \
		static void fillName(std::stringstream& ss) { \
			ss << "_" << #CLASSNAME << "_1"; \
			E::fillName(ss); \
		} \
 \
		void setCudaParam(CUfunction& cuFunction, int& offset) const { \
			_l.setCudaParam(cuFunction, offset); \
			\
			CUresult result; \
			ALIGN_UP(offset, __alignof(_r)); \
			result = cuParamSetf(cuFunction, offset, _r); \
			assert(result == CUDA_SUCCESS); \
			offset += sizeof(_r); \
		}; \
 \
}; \
 \
template<class E> \
CLASSNAME<E> operator OPSYMBOL( Expression<E> const &l, float const &r) { \
	return CLASSNAME<E> (l, r); \
} \
//}}}

// {{{ make_binary_scalar_expression_function
#define make_binary_scalar_expression_function(CLASSNAME, FUNCTIONNAME, CUDAFUNCTIONNAME) \
template <typename E> \
class CLASSNAME : public Expression<CLASSNAME<E> > { \
	public: \
		float const& _l; \
		E const& _r; \
 \
		CLASSNAME( float const& l, Expression<E> const& r) \
			: _l(l), _r(r) { } \
 \
		unsigned int size() const { return _r.size(); } \
 \
		static void fillCudaStrings(char& current_var, std::stringstream& eval_line, \
				std::stringstream& def_line) {  \
			eval_line << CUDAFUNCTIONNAME << "(" << current_var; \
			eval_line	<< ", "; \
			def_line << "float " << current_var++ << ", "; \
			E::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< ")"; \
		} \
 \
		static void fillName(std::stringstream& ss) { \
			ss << "_" << #CLASSNAME << "_1"; \
			E::fillName(ss); \
		} \
 \
		void setCudaParam(CUfunction& cuFunction, int& offset) const { \
			 \
			CUresult result; \
			ALIGN_UP(offset, __alignof(_l)); \
			result = cuParamSetf(cuFunction, offset, _l); \
			assert(result == CUDA_SUCCESS); \
			offset += sizeof(_l); \
 \
			_r.setCudaParam(cuFunction, offset); \
		}; \
 \
}; \
 \
template<class E> \
CLASSNAME<E> FUNCTIONNAME( float const &l, Expression<E> const &r) { \
	return CLASSNAME<E> (l, r); \
}
//}}}

// {{{ make_binary_expression_scalar_function
#define make_binary_expression_scalar_function(CLASSNAME, FUNCTIONNAME, CUDAFUNCTIONNAME) \
template <typename E> \
class CLASSNAME : public Expression<CLASSNAME<E> > { \
	public: \
		E const& _l; \
		float const& _r; \
 \
		CLASSNAME( Expression<E> const& l, float const& r) \
			: _l(l), _r(r) { } \
 \
		unsigned int size() const { return _l.size(); } \
 \
		static void fillCudaStrings(char& current_var, std::stringstream& eval_line, \
				std::stringstream& def_line) {  \
			eval_line << CUDAFUNCTIONNAME << "("; \
			E::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< ", " << current_var << ")"; \
			def_line << ", float " << current_var++ ; \
		} \
 \
		static void fillName(std::stringstream& ss) { \
			ss << "_" << #CLASSNAME << "_1"; \
			E::fillName(ss); \
		} \
 \
		void setCudaParam(CUfunction& cuFunction, int& offset) const { \
			_l.setCudaParam(cuFunction, offset); \
			 \
			CUresult result; \
			ALIGN_UP(offset, __alignof(_r)); \
			result = cuParamSetf(cuFunction, offset, _r); \
			assert(result == CUDA_SUCCESS); \
			offset += sizeof(_r); \
		}; \
 \
}; \
\
template<class E> \
CLASSNAME<E> FUNCTIONNAME( Expression<E> const &l, float const &r) { \
	return CLASSNAME<E> (l, r); \
}
//}}}

//{{{ make_binary_function
#define make_binary_function(CLASSNAME, FUNCTIONNAME, CUDAFUNCTION) \
template <typename E1, typename E2> \
class CLASSNAME : public Expression<CLASSNAME<E1, E2> > { \
	public: \
		E1 const& _l; \
		E2 const& _r; \
\
		CLASSNAME( Expression<E1> const& l, Expression<E2> const& r) \
			: _l(l), _r(r) { assert(_l.size() == _r.size()); } \
\
		unsigned int size() const { return _r.size(); } \
\
		static void fillCudaStrings(char& current_var, std::stringstream& eval_line, \
				std::stringstream& def_line) {  \
			eval_line << CUDAFUNCTION << "("; \
			E1::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< ", " ; \
			def_line << ", "; \
			E2::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< ")"; \
		} \
\
		static void fillName(std::stringstream& ss) { \
			ss << "_" << #CLASSNAME << "_2"; \
			E1::fillName(ss); \
			E2::fillName(ss); \
		} \
 \
		void setCudaParam(CUfunction& cuFunction, int& offset) const { \
			_l.setCudaParam(cuFunction, offset); \
			_r.setCudaParam(cuFunction, offset); \
		}; \
\
}; \
\
template<class E1, class E2> \
CLASSNAME<E1, E2> inline FUNCTIONNAME(Expression<E1> const &l,\
		Expression<E2> const &r) { \
	return CLASSNAME<E1, E2> (l, r); \
}
//}}}

//{{{ make_binary_infix_operator
#define make_binary_infix_operator(CLASSNAME, OPSYMBOL, CUDAOPSYMBOL) \
template <typename E1, typename E2> \
class CLASSNAME : public Expression<CLASSNAME<E1, E2> > { \
	public: \
		E1 const& _l; \
		E2 const& _r; \
\
		CLASSNAME( Expression<E1> const& l, Expression<E2> const& r) \
			: _l(l), _r(r) { assert(_l.size() == _r.size()); } \
\
		unsigned int size() const { return _r.size(); } \
\
		static void fillCudaStrings(char& current_var, std::stringstream& eval_line, \
				std::stringstream& def_line) {  \
			eval_line << "("; \
			E1::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< " " << CUDAOPSYMBOL << " "; \
			def_line << ", "; \
			E2::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< ")"; \
		} \
\
		static void fillName(std::stringstream& ss) { \
			ss << "_" << #CLASSNAME << "_2"; \
			E1::fillName(ss); \
			E2::fillName(ss); \
		} \
 \
		void setCudaParam(CUfunction& cuFunction, int& offset) const { \
			_l.setCudaParam(cuFunction, offset); \
			_r.setCudaParam(cuFunction, offset); \
		}; \
\
}; \
\
template<class E1, class E2> \
CLASSNAME<E1, E2> inline operator OPSYMBOL(Expression<E1> const &l,\
		Expression<E2> const &r) { \
	return CLASSNAME<E1, E2> (l, r); \
}
//}}}

// {{{ make_unary_function
#define make_unary_function(CLASSNAME, FUNCTIONNAME, CUDAFUNCTIONNAME) \
template <typename E> \
class CLASSNAME : public Expression<CLASSNAME<E> > { \
	public: \
		E const& _r; \
 \
		CLASSNAME( Expression<E> const& r) \
			: _r(r) { } \
 \
		unsigned int size() const { return _r.size(); } \
 \
		static void fillCudaStrings(char& current_var, std::stringstream& eval_line, \
				std::stringstream& def_line) {  \
			eval_line << CUDAFUNCTIONNAME << "(" ; \
			E::fillCudaStrings(current_var, eval_line, def_line); \
			eval_line	<< ")"; \
		} \
 \
		static void fillName(std::stringstream& ss) { \
			ss << "_" << #CLASSNAME << "_1"; \
			E::fillName(ss); \
		} \
 \
		void setCudaParam(CUfunction& cuFunction, int& offset) const { \
			_r.setCudaParam(cuFunction, offset); \
		}; \
 \
}; \
 \
template<class E> \
CLASSNAME<E> FUNCTIONNAME( Expression<E> const &r) { \
	return CLASSNAME<E> (r); \
} \
//}}}

//{{{ make_cublas macros

#define make_cublas_two_args(FUNCTIONNAME, RETURNTYPE, CUBLASFUNCTION) \
template<typename E1, typename E2> \
RETURNTYPE FUNCTIONNAME(const Expression<E1>& l, const Expression<E2>& r) {\
	cudaVec tmp_l(l);\
	cudaVec tmp_r(r);\
	std::cerr << "evaluate cublas function with copy left and copy right " << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(tmp_l.size(), (float*) tmp_l.data().get(), 1, (float*) tmp_r.data().get(), 1);\
}\
template<typename E2> \
RETURNTYPE FUNCTIONNAME(const cudaVec& l, const Expression<E2>& r) {\
	cudaVec tmp_r(r);\
	std::cerr << "evaluate cublas function with copy right " << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(l.size(), (float*) l.data().get(), 1, (float*) tmp_r.data().get(), 1);\
}\
template<typename E1> \
RETURNTYPE FUNCTIONNAME(const Expression<E1>& l, const cudaVec& r) {\
	cudaVec tmp_l(l);\
	std::cerr << "evaluate cublas function with copy left " << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(tmp_l.size(), (float*) tmp_l.data().get(), 1, (float*) r.data().get(), 1);\
}\
inline RETURNTYPE FUNCTIONNAME(const cudaVec& l, const cudaVec& r) {\
	std::cerr << "evaluate cublas function without copy " << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(l.size(), (float*) l.data().get(), 1, (float*) r.data().get(), 1);\
}

#define make_cublas_infix_operator(OPSYMBOL, RETURNTYPE, CUBLASFUNCTION) \
template<typename E1, typename E2> \
RETURNTYPE operator OPSYMBOL(const Expression<E1>& l, const Expression<E2>& r) {\
	cudaVec tmp_l(l);\
	cudaVec tmp_r(r);\
	std::cerr << "evaluate cublas function with copy left and copy right " << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(tmp_l.size(), (float*) tmp_l.data().get(), 1, (float*) tmp_r.data().get(), 1);\
}\
template<typename E2> \
RETURNTYPE operator OPSYMBOL(const cudaVec& l, const Expression<E2>& r) {\
	cudaVec tmp_r(r);\
	std::cerr << "evaluate cublas function with copy right" << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(l.size(), (float*) l.data().get(), 1, (float*) tmp_r.data().get(), 1);\
}\
template<typename E1> \
RETURNTYPE operator OPSYMBOL(const Expression<E1>& l, const cudaVec& r) {\
	cudaVec tmp_l(l);\
	std::cerr << "evaluate cublas function with copy left" << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(tmp_l.size(), (float*) tmp_l.data().get(), 1, (float*) r.data().get(), 1);\
}\
inline RETURNTYPE operator OPSYMBOL(const cudaVec& l, const cudaVec& r) {\
	std::cerr << "evaluate cublas function without copy " << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(l.size(), (float*) l.data().get(), 1, (float*) r.data().get(), 1);\
}

#define make_cublas_one_arg(FUNCTIONNAME, RETURNTYPE, CUBLASFUNCTION) \
template<typename E> \
RETURNTYPE FUNCTIONNAME(const Expression<E>& l) {\
	cudaVec tmp_l(l);\
	std::cerr << "evaluate cublas function with copy " << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(tmp_l.size(), (float*) tmp_l.data().get(), 1);\
}\
inline RETURNTYPE FUNCTIONNAME(const cudaVec& l) {\
	std::cerr << "evaluate cublas function without copy " << #CUBLASFUNCTION << "()" << std::endl;\
	return CUBLASFUNCTION(l.size(), (float*) l.data().get(), 1);\
}
//}}}

// {{{ define 
make_binary_infix_operator(VecInfixAddExpression, +, "+"); // component-by-component addition
make_binary_scalar_expression_infix_operator(ScVecInfixAddExpression, +, "+");
make_binary_expression_scalar_infix_operator(VecScInfixAddExpression, +, "+");

make_binary_infix_operator(VecInfixSubExpression, -, "-"); // component-by-component subtraction
make_binary_expression_scalar_infix_operator(VecScInfixSubExpression, -, "-");
make_binary_scalar_expression_infix_operator(ScVecInfixSubExpression, -, "-");

make_binary_infix_operator(VecInfixMulExpression, *, "*"); // component-by-component multiplication
make_binary_scalar_expression_infix_operator(ScVecInfixMulExpression, *, "*"); 
make_binary_expression_scalar_infix_operator(VecScInfixMulExpression, *, "*");

make_binary_infix_operator(VecInfixDivExpression, /, "/"); // component-by-component division
make_binary_expression_scalar_infix_operator(VecScInfixDivExpression, /, "/");
make_binary_scalar_expression_infix_operator(ScVecInfixDivExpression, /, "/");

/** These Expressions don't work on floats
 * make_binary_infix_operator(VecInfixModExpression, %, "%");
 * make_binary_expression_scalar_infix_operator(VecScInfixModExpression, %, "%");
 * make_binary_scalar_expression_infix_operator(ScVecInfixModExpression, %, "%");
 * 
 * make_binary_infix_operator(VecInfixBitXorExpression, ^, "^");
 * make_binary_expression_scalar_infix_operator(VecScInfixBitXorExpression, ^, "^");
 * make_binary_scalar_expression_infix_operator(ScVecInfixBitXorExpression, ^, "^");
 * 
 * make_binary_infix_operator(VecInfixBitAndExpression, &, "&");
 * make_binary_expression_scalar_infix_operator(VecScInfixBitAndExpression, &, "&");
 * make_binary_scalar_expression_infix_operator(ScVecInfixBitAndExpression, &, "&");
 * 
 * make_binary_infix_operator(VecInfixBitOrExpression, |, "|");
 * make_binary_expression_scalar_infix_operator(VecScInfixBitOrExpression, |, "|");
 * make_binary_scalar_expression_infix_operator(ScVecInfixBitOrExpression, |, "|");
 * 
 * make_binary_infix_operator(VecInfixShiftLeftExpression, <<, "<<");
 * make_binary_expression_scalar_infix_operator(VecScInfixShiftLeftExpression, <<, "<<");
 * make_binary_scalar_expression_infix_operator(ScVecInfixShiftLeftExpression, <<, "<<");
 * 
 * make_binary_infix_operator(VecInfixShiftRightExpression, >>, ">>");
 * make_binary_expression_scalar_infix_operator(VecScInfixShiftRightExpression, >>, ">>");
 * make_binary_scalar_expression_infix_operator(ScVecInfixShiftRightExpression, >>, ">>");
 */
make_binary_infix_operator(VecInfixAndExpression, &&, "&&");
make_binary_expression_scalar_infix_operator(VecScInfixAndExpression, &&, "&&");
make_binary_scalar_expression_infix_operator(ScVecInfixAndExpression, &&, "&&");

make_binary_infix_operator(VecInfixOrExpression, ||, "||");
make_binary_expression_scalar_infix_operator(VecScInfixOrExpression, ||, "||");
make_binary_scalar_expression_infix_operator(ScVecInfixOrExpression, ||, "||");

make_binary_infix_operator(VecInfixEqualToExpression, ==, "==");
make_binary_expression_scalar_infix_operator(VecScInfixEqualToExpression, ==, "==");
make_binary_scalar_expression_infix_operator(ScVecInfixEqualToExpression, ==, "==");

make_binary_infix_operator(VecInfixNotEqualToExpression, !=, "!=");
make_binary_expression_scalar_infix_operator(VecScInfixNotEqualToExpression, !=, "!=");
make_binary_scalar_expression_infix_operator(ScVecInfixNotEqualToExpression, !=, "!=");

make_binary_infix_operator(VecInfixLessThenExpression, <, "<");
make_binary_expression_scalar_infix_operator(VecScInfixLessThenExpression, <, "<");
make_binary_scalar_expression_infix_operator(ScVecInfixLessThenExpression, <, "<");

make_binary_infix_operator(VecInfixGreaterThenExpression, >, ">");
make_binary_expression_scalar_infix_operator(VecScInfixGreaterThenExpression, >, ">");
make_binary_scalar_expression_infix_operator(ScVecInfixGreaterThenExpression, >, ">");

make_binary_infix_operator(VecInfixLessThenOrEqualToExpression, <=, "<=");
make_binary_expression_scalar_infix_operator(VecScInfixLessThenOrEqualToExpression, <=, "<=");
make_binary_scalar_expression_infix_operator(ScVecInfixLessThenOrEqualToExpression, <=, "<=");

make_binary_infix_operator(VecInfixGreaterThenOrEqualToExpression, >=, ">=");
make_binary_expression_scalar_infix_operator(VecScInfixGreaterThenOrEqualToExpression, >=, ">=");
make_binary_scalar_expression_infix_operator(ScVecInfixGreaterThenOrEqualToExpression, >=, ">=");

// returns a vector, with the component-by-component absolute value of a vector
make_unary_function(VecUnaryMaxExpression, abs, "abs");

// returns a vector with the computed sinus for each component.
make_unary_function(VecUnarySinExpression, sin, "sin");
// returns a vector with the computed cosinus for each component.
make_unary_function(VecUnaryCosExpression, cos, "cos");
// returns a vector with the computed tan for each component.
make_unary_function(VecUnaryTanExpression, tan, "tan");
// returns a vector with the computed arcus sinus for each component.
make_unary_function(VecUnaryArcSinExpression, asin, "asin");
// returns a vector with the computed arcus cosinus for each component.
make_unary_function(VecUnaryArcCosExpression, acos, "acos");
// returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryArcTanExpression, atan, "atan");

// returns a vector with the computed sinus for each component.
make_unary_function(VecUnarySinHExpression, sinh, "sinh");

// returns a vector with the computed cosinus for each component.
make_unary_function(VecUnaryCosHExpression, cosh, "cosh");

// returns a vector with the computed tan for each component.
make_unary_function(VecUnaryTanHExpression, tanh, "tanh");

// returns a vector with the computed arcus sinus for each component.
make_unary_function(VecUnaryArcSinHExpression, asinh, "asinh");

// returns a vector with the computed arcus cosinus for each component.
make_unary_function(VecUnaryArcCosHExpression, acosh, "acosh");

// returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryArcTanHExpression, atanh, "atanh");
//
// returns a vector with the computed squareroot for each component.
make_unary_function(VecUnarySqrtExpression, sqrt, "sqrt");

//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryExp2Expression, exp2, "exp2");
//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryExp10Expression, exp10, "exp10");
//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryExpExpression, exp, "exp");
//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryLog2Expression, log2, "log2");
//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryLog10Expression, log10, "log10");
//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryLogExpression, log, "log");
//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryTruncExpression, trunc, "trunc");
//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryFloorExpression, floor, "floor");
//TODO returns a vector with the computed arcus tan for each component.
make_unary_function(VecUnaryRoundExpression, round, "round");

make_unary_function(VecUnaryIsNanExpression, isNan, "isnan");
make_unary_function(VecUnaryIsInfExpression, isInf, "isin");
make_unary_function(VecUnarySignBitExpression, signBit, "signbit");

// returns a vector, with the component-by-component minimum of two vectors
make_binary_function(VecBinaryMinExpression, min, "min");
// returns a vector, with the component-by-component minimum of a vector and a scalar.
make_binary_scalar_expression_function(ScVecBinaryMinExpression, min, "min");
make_binary_expression_scalar_function(VecScBinaryMinExpression, min, "min");

// returns a vector, with the component-by-component maximux of two vectors
make_binary_function(VecBinaryMaxExpression, max, "max");
// returns a vector, with the component-by-component maximum of a vector and a scalar.
make_binary_scalar_expression_function(ScVecBinaryMaxExpression, max, "max");
make_binary_expression_scalar_function(VecScBinaryMaxExpression, max, "max");

// returns a vector, with the component-by-component maximux of two vectors
make_binary_function(VecBinaryPowExpression, pow, "pow");
// returns a vector, with the component-by-component maximum of a vector and a scalar.
make_binary_scalar_expression_function(ScVecBinaryPowExpression, pow, "pow");
make_binary_expression_scalar_function(VecScBinaryPowExpression, pow, "pow");






// returns the summation of all components.
make_cublas_one_arg(sum, float, cublasSasum);

//finds the smallest index of a maximum magnitude element
make_cublas_one_arg(argmax, int, cublasIsamax);
//finds the smallest index of a minimum magnitude element
make_cublas_one_arg(argmin, int, cublasIsamin);

//computes the Euclidean norm of a vector
make_cublas_one_arg(norm2, float, cublasSnrm2);

//computes the dot product of two vectors
make_cublas_two_args(dot, float, cublasSdot);
//}}}

} // namespace cudaVector
#endif //_CUDA_VECTOR_H_



/* further CUDA functions: see CUDAINCLUDE/math_functions.h
 *
 * rsqrt
 * expm1
 * log1p
 * ldexp
 * logb
 * ilogb
 * rint
 * nearbyint
 * ceil
 * fdim
 * atan2
 * hypot
 * cbrt
 * rcbrt
 * sinpi
 * modf
 * fmod
 * remainder
 * remquo
 * erf
 * erfinv
 * erfc
 * erfcinv
 * tgamma
 * lgamma
 * copysignf
 * nextafterf
 * finitef
 * 
 */
