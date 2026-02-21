/*Standard single solve*/

#include <cstdint>
#include <memory>
#include <vector>
#include <complex>
#include <type_traits>
// #include <cuComplex.h> // For device-side complex number operations - not needed now that I am not writing kernels

#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "cudss.h"
#include <mutex>
#include <unordered_map>

namespace ffi = xla::ffi;
namespace nb = nanobind;

// verification ================================================================
#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            return ffi::Error::Success(); \
        } \
    } while(0);

#define CUDA_CHECK(call)                                       \
  do {                                                         \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
      printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__,   \
             cudaGetErrorString(err));                         \
      return ffi::Error::Internal("A CUDA call failed.");      \
    }                                                          \
  } while (0)


// Debugging functions =========================================================
template <typename T>
void print_device_data(
    const char* label,
    void* device_ptr,
    size_t n_batch,
    size_t n_elements_per_batch)
{
    // Ensure we have a valid pointer and something to print
    if (!device_ptr || n_batch == 0 || n_elements_per_batch == 0) return;

    std::cout << "\n--- Debug Print: " << label << " ---" << std::endl;

    // Calculate total size and create a host-side vector
    size_t total_elements = n_batch * n_elements_per_batch;
    std::vector<T> host_data(total_elements);

    // Copy all data from GPU to CPU in one go
    cudaMemcpy(
        host_data.data(),
        device_ptr,
        total_elements * sizeof(T),
        cudaMemcpyDeviceToHost
    );

    // Loop through each batch and print its contents
    for (size_t i = 0; i < n_batch; ++i) {
        std::cout << "Batch " << i << ": [";
        size_t batch_start_index = i * n_elements_per_batch;
        for (size_t j = 0; j < n_elements_per_batch; ++j) {
            std::cout << host_data[batch_start_index + j];
            if (j < n_elements_per_batch - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}

// Helper function for data types ==============================================
template <ffi::DataType T> cudaDataType get_cuda_data_type();
template<> cudaDataType get_cuda_data_type<ffi::F32>() { return CUDA_R_32F; }
template<> cudaDataType get_cuda_data_type<ffi::F64>() { return CUDA_R_64F; }
template<> cudaDataType get_cuda_data_type<ffi::C64>() { return CUDA_C_32F; }
template<> cudaDataType get_cuda_data_type<ffi::C128>() { return CUDA_C_64F; }

template <ffi::DataType T>
struct get_native_data_type;
template<> struct get_native_data_type<ffi::F32> { using type = float; };
template<> struct get_native_data_type<ffi::F64> { using type = double; };
template<> struct get_native_data_type<ffi::C64> { using type = std::complex<float>; };
template<> struct get_native_data_type<ffi::C128> { using type = std::complex<double>; };

// Structure definitions =======================================================
template <ffi::DataType T>
struct CudssSharedState {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t x = nullptr;
    cudssMatrix_t b = nullptr;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    typename get_native_data_type<T>::type* diag_temp = nullptr; // temporary storage for diagonal values
    int32_t* perm_temp = nullptr; // temporary storage for permutation
    int64_t n;
    int64_t nnz;
    int64_t nrhs;
    int64_t call_count = 0; // necessary for detecting if we need further instantiation in execution stage
    size_t sizeWritten;
    int32_t do_refactorize; // host-side flag read from traced GPU signal
    int32_t do_solve; // host-side flag: 1 = solve with IR, 0 = solve without IR
    int32_t ir_nsteps; // host-side value read from traced GPU signal
    cudaDataType cuda_dtype = get_cuda_data_type<T>();

    // this is literally only for debugging
    using native_dtype = typename get_native_data_type<T>::type;

    ~CudssSharedState() {
        // TODO: Fix destructor - currently causes double free on CUDA 13 / cuDSS 0.7
        // if (handle) {
        //     cudssMatrixDestroy(A);
        //     cudssMatrixDestroy(b);
        //     cudssMatrixDestroy(x);
        //     cudssDataDestroy(handle, data);
        //     cudssConfigDestroy(config);
        //     cudssDestroy(handle);
        // }
    }
};

// Thin wrapper that XLA FFI State<T> manages; points to shared state via registry
template <ffi::DataType T>
struct CudssState {
    static xla::ffi::TypeId id;
    std::shared_ptr<CudssSharedState<T>> shared;
};

template <> ffi::TypeId CudssState<ffi::F32>::id = {};
template <> ffi::TypeId CudssState<ffi::F64>::id = {};
template <> ffi::TypeId CudssState<ffi::C64>::id = {};
template <> ffi::TypeId CudssState<ffi::C128>::id = {};

// Global registry: all custom_calls from the same Python CuDSSSolver share one CudssSharedState
template <ffi::DataType T>
struct CudssRegistry {
    static std::mutex mtx;
    static std::unordered_map<int64_t, std::shared_ptr<CudssSharedState<T>>> states;
    static std::shared_ptr<CudssSharedState<T>> get_or_create(int64_t solver_id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = states.find(solver_id);
        if (it != states.end()) return it->second;
        auto s = std::make_shared<CudssSharedState<T>>();
        states[solver_id] = s;
        return s;
    }
};
template <ffi::DataType T> std::mutex CudssRegistry<T>::mtx;
template <ffi::DataType T> std::unordered_map<int64_t, std::shared_ptr<CudssSharedState<T>>> CudssRegistry<T>::states;

// instantiation ===============================================================

// instantiate everything that is not a function of the context (cudaStream_t)
template <ffi::DataType T>
static ffi::ErrorOr<std::unique_ptr<CudssState<T>>> CudssInstantiate(
    const int64_t solver_id,                // unique id for shared state registry
    const int64_t device_id,                // the device to run this on
    const int64_t mtype_id,                 // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                  // {0: full, 1: triu, 2: tril}
) {

    // Look up or create shared state from the global registry
    auto shared = CudssRegistry<T>::get_or_create(solver_id);

    // Create the thin wrapper that XLA will manage
    auto wrapper = std::make_unique<CudssState<T>>();
    wrapper->shared = shared;

    // Only initialize mtype/mview if this is the first instantiation (handle still null)
    if (shared->handle == nullptr) {
        // check on the type of matrix being solved
        if (mtype_id == 0) {
            shared->mtype = CUDSS_MTYPE_GENERAL;
        } else if (mtype_id == 1) {
            shared->mtype = CUDSS_MTYPE_SYMMETRIC;
        } else if (mtype_id == 2) {
            shared->mtype = CUDSS_MTYPE_HERMITIAN;
        } else if (mtype_id == 3) {
            shared->mtype = CUDSS_MTYPE_SPD;
        } else if (mtype_id == 4) {
            shared->mtype = CUDSS_MTYPE_HPD;
        } else {
            throw std::invalid_argument("Invalid mtype_id. Valid options: 0: general, 1: symmetric, 2: hermitian, 3: spd, 4: hpd");
        }

        // check on the view of the matrix provided
        if (mview_id == 0) {
            shared->mview = CUDSS_MVIEW_FULL;
        } else if (mview_id == 1) {
            shared->mview = CUDSS_MVIEW_UPPER;
        } else if (mview_id == 2) {
            shared->mview = CUDSS_MVIEW_LOWER;
        } else {
            throw std::invalid_argument("Invalid mview_id. Valid options: 0: full, 1: upper, 2: lower");
        }

        shared->nrhs = 1; // the non-batched case

        // CUDA setup
        cudaSetDevice(device_id);

        // Allocate temporary storage for diagonal and permutation
        cudaMalloc(&shared->diag_temp, shared->n * sizeof(typename get_native_data_type<T>::type));
        cudaMalloc(&shared->perm_temp, shared->n * sizeof(int32_t));
    }

    return ffi::ErrorOr<std::unique_ptr<CudssState<T>>>(std::move(wrapper));
}

// execution ===================================================================
template <ffi::DataType T>
static ffi::Error CudssExecute(
    cudaStream_t stream,                    // JAXs stream given to this context (jit)
    CudssState<T>* wrapper,                 // the thin wrapper we instantiated in CudssInstantiate
    ffi::Buffer<T> b_values_buf,            // the real input data that varies per solution
    ffi::Buffer<T> csr_values_buf,          // the real input data that varies per solution
    ffi::Buffer<ffi::S32> offsets_buf,
    ffi::Buffer<ffi::S32> columns_buf,
    ffi::Buffer<ffi::S32> refactorize_signal,      // whether we should refactorize within jit
    ffi::Buffer<ffi::S32> solve_signal,      // whether we should solve within jit
    ffi::Buffer<ffi::S32> ir_nsteps_signal,  // number of iterative refinement steps
    ffi::ResultBuffer<T> out_values_buf,    // the output buffer we write the answer to
    ffi::ResultBuffer<T> diag_buf,          // the output buffer we write the answer to
    ffi::ResultBuffer<ffi::S32> perm_buf,   // the output buffer we write the answer to
    const int64_t solver_id,                // unique id for shared state registry
    const int64_t device_id,                // the device to run this on
    const int64_t mtype_id,                 // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                  // {0: full, 1: triu, 2: tril}
) {

    // Dereference shared state from the wrapper
    auto* state = wrapper->shared.get();

    // Synchronize with XLA stream to ensure signal buffers have been written
    cudaStreamSynchronize(stream);

    cudaMemcpy(&state->do_solve, solve_signal.typed_data(),
                sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state->ir_nsteps, ir_nsteps_signal.typed_data(),
                sizeof(int32_t), cudaMemcpyDeviceToHost);

    // instantiate system branch
    if (state->call_count == 0) {

        // figure this out on first call
        state->n = offsets_buf.element_count() - 1;
        state->nnz = columns_buf.element_count();

        // CuDSS setup
        CUDSS_CALL_AND_CHECK(cudssCreate(&state->handle), state->status, "cudssCreate");
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&state->config), state->status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(state->handle, &state->data), state->status, "cudssDataCreate");

        // CuDSS structures creation
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->b, state->n, state->nrhs, state->n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for b");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->x, state->n, state->nrhs, state->n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for x");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&state->A, state->n, state->n, state->nnz,
            offsets_buf.typed_data(), NULL,
            columns_buf.typed_data(),
            csr_values_buf.typed_data(),
            CUDA_R_32I, state->cuda_dtype,
            state->mtype, state->mview, state->base), state->status, "cudssMatrixCreateCsr");

        // CuDSS config - iterative refinement steps from runtime signal
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_IR_N_STEPS,
                            &state->ir_nsteps, sizeof(state->ir_nsteps)), state->status, "cudssConfigSet ir_nsteps");

        // cold solve - analyze, factorize, solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute analysis");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");

        if (state->do_solve) {
            CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
                state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");
        }
        state->call_count++;
    }
    else {
        // stream can change between calls!!!
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");

        // Read refactorize signal from GPU to host (cudaMemcpy is synchronous)
        cudaMemcpy(&state->do_refactorize, refactorize_signal.typed_data(),
                   sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Always update b and x pointers to valid XLA buffers before any cuDSS phase
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->b, b_values_buf.typed_data()), state->status, "update_pointers b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->x, out_values_buf->typed_data()), state->status, "update_pointers x");

        // Always update A values pointer so IR uses correct matrix for residuals
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->A, csr_values_buf.typed_data()), state->status, "update_pointers A");

        // Conditionally refactorize based on traced signal
        if (state->do_refactorize) {
            CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_REFACTORIZATION,
                state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");
        }

        // Update IR steps from runtime signal before solve
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_IR_N_STEPS,
                            &state->ir_nsteps, sizeof(state->ir_nsteps)), state->status, "cudssConfigSet ir_nsteps");

        // Conditionally solve based on traced signal
        if (state->do_solve) {
            CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
                state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");
        }
    }

    CUDSS_CALL_AND_CHECK(cudssDataGet(state->handle, state->data, CUDSS_DATA_DIAG, diag_buf->typed_data(),
                    state->n * sizeof(typename get_native_data_type<T>::type), &state->sizeWritten), state->status, "cudssDataGet DATA_DIAG");
    CUDSS_CALL_AND_CHECK(cudssDataGet(state->handle, state->data, CUDSS_DATA_PERM_REORDER_ROW, perm_buf->typed_data(),
                    state->n * sizeof(int32_t), &state->sizeWritten), state->status, "cudssDataGet DATA_PERM_REORDER_ROW");

    return ffi::Error::Success();
}

// minimize XLA/nanobind boilerplate with a couple macros ======================

// XLA ffi handler definitions for all datatypes
#define DEFINE_CUDSS_FFI_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kCudssInstantiate##TypeName, CudssInstantiate<DataType>, \
        ffi::Ffi::BindInstantiate() \
            .Attr<int64_t>("solver_id") \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kCudssExecute##TypeName, CudssExecute<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Ctx<ffi::State<CudssState<DataType>>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<int64_t>("solver_id") \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id"));

// Generate all the FFI handlers using the macro
DEFINE_CUDSS_FFI_HANDLERS(f32, ffi::F32);
DEFINE_CUDSS_FFI_HANDLERS(f64, ffi::F64);
DEFINE_CUDSS_FFI_HANDLERS(c64, ffi::C64);
DEFINE_CUDSS_FFI_HANDLERS(c128, ffi::C128);

// nanobind module exporting macro
#define EXPORT_CUDSS_HANDLERS(m, TypeName, DataType) \
    m.def("type_id_" #TypeName, []() { \
        return nb::capsule(reinterpret_cast<void*>(&CudssState<DataType>::id)); \
    }); \
    m.def("handler_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecute##TypeName)); \
        return d; \
    });

// generate all nanobind modules! :)
NB_MODULE(single_solve, m) {
    EXPORT_CUDSS_HANDLERS(m, f32, ffi::F32);
    EXPORT_CUDSS_HANDLERS(m, f64, ffi::F64);
    EXPORT_CUDSS_HANDLERS(m, c64, ffi::C64);
    EXPORT_CUDSS_HANDLERS(m, c128, ffi::C128);
}

