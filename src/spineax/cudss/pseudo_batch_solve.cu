/*
spineax's cuDSS backend: token-based persistent factorization
(docs/token_design.md).

Batch systems are solved WITHOUT cuDSS's batch API: a batch of B systems is
its block-diagonal concatenation, one bigger sparse system. This exposes data
(per-block inertia via diag/perm) the batch API does not, and makes a single
solve just the batch_size=1 special case — so this one module is the whole
backend.

(This file is .cu so cmake sees it needs nvcc for the block-structure
expansion kernels.)
*/

#include <cstdint>
#include <memory>
#include <vector>
#include <complex>
#include <type_traits>
// #include <cuComplex.h> // For device-side complex number operations

#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "cudss.h"
#include <mutex>
#include <map>
#include <unordered_map>
#include <atomic>
#include <cstdlib>
#include <list>
#include <string>

namespace ffi = xla::ffi;
namespace nb = nanobind;

// Helper function for data types ==============================================
template <ffi::DataType T> cudssDataType_t get_cudss_data_type();
template<> cudssDataType_t get_cudss_data_type<ffi::F32>() { return CUDSS_R_32F; }
template<> cudssDataType_t get_cudss_data_type<ffi::F64>() { return CUDSS_R_64F; }
template<> cudssDataType_t get_cudss_data_type<ffi::C64>() { return CUDSS_C_32F; }
template<> cudssDataType_t get_cudss_data_type<ffi::C128>() { return CUDSS_C_64F; }

template <ffi::DataType T>
struct get_native_data_type;
template<> struct get_native_data_type<ffi::F32> { using type = float; };
template<> struct get_native_data_type<ffi::F64> { using type = double; };
template<> struct get_native_data_type<ffi::C64> { using type = std::complex<float>; };
template<> struct get_native_data_type<ffi::C128> { using type = std::complex<double>; };

// execution ===================================================================

// GPU kernel to create batched column indices in parallel
__global__ void create_batched_columns_kernel(
    const int32_t* __restrict__ single_columns,
    int32_t* __restrict__ batched_columns,
    int64_t nnz, int64_t n, int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * nnz;

    if (idx < total_elements) {
        int64_t batch_idx = idx / nnz;          // Which batch
        int64_t col_idx = idx % nnz;            // Which element within batch
        int32_t column_offset = batch_idx * n;  // Offset for this batch

        batched_columns[idx] = single_columns[col_idx] + column_offset;
    }
}

// GPU kernel to create batched row offsets in parallel
__global__ void create_batched_offsets_kernel(
    const int32_t* __restrict__ single_offsets,
    int32_t* __restrict__ batched_offsets,
    int64_t n, int64_t nnz, int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * n + 1;

    if (idx < total_elements) {
        if (idx == batch_size * n) {
            // Last element: total nnz
            batched_offsets[idx] = batch_size * nnz;
        } else {
            int64_t batch_idx = idx / n;        // Which batch
            int64_t row_idx = idx % n;          // Which row within batch
            int32_t nnz_offset = batch_idx * nnz; // Offset for this batch

            batched_offsets[idx] = single_offsets[row_idx] + nnz_offset;
        }
    }
}

// GPU-accelerated function to populate batched CSR structure
// Note: Memory must be pre-allocated before calling this function
static void create_batched_csr_structure(
    int32_t* single_offsets_ptr, int32_t* single_columns_ptr,
    int64_t n, int64_t nnz, int64_t batch_size,
    int32_t** batched_offsets_ptr, int32_t** batched_columns_ptr,
    cudaStream_t stream = 0)
{
    // Launch kernels to populate batched arrays from single pattern
    const int threads = 256;

    // Launch kernel to create batched columns
    int64_t total_columns = batch_size * nnz;
    int blocks_columns = (total_columns + threads - 1) / threads;
    create_batched_columns_kernel<<<blocks_columns, threads, 0, stream>>>(
        single_columns_ptr, *batched_columns_ptr, nnz, n, batch_size);

    // Launch kernel to create batched offsets
    int64_t total_offsets = batch_size * n + 1;
    int blocks_offsets = (total_offsets + threads - 1) / threads;
    create_batched_offsets_kernel<<<blocks_offsets, threads, 0, stream>>>(
        single_offsets_ptr, *batched_offsets_ptr, n, nnz, batch_size);

    // No synchronization needed here - stream will handle dependencies
}


// =============================================================================
// Token-based persistent factorization (docs/token_design.md)
//
// This is THE token implementation: a single solve is just the batch_size=1
// special case of the block-diagonal construction, so all four tiers live
// here, over one process-global LRU registry:
//
//   analyze(values(B·nnz), offsets, columns)  -> token       block ANALYSIS
//   factorize(tokens, values, ir)  -> tokens                 block FACTORIZATION
//   refactorize(tokens, values, ir)-> tokens                 block REFACTORIZATION
//   solve(tokens, b, ir)           -> x                      block SOLVE
//   query(tokens)                  -> every cuDSS data item
//
// Phase handlers are pure phase execution; query is the ONE door to
// post-factorization data (diag, permutations, inertia inputs, ...), returned
// unconditionally with Python figuring out what it wants — this subsumes
// single_solve_re.cpp. solve derives nrhs from element_count / (batch·n),
// which makes multi-RHS on a single entry and a block solve on a batch entry
// the same code path.
//
// The token buffer may carry ONE id (explicit door) or B equal copies (minted
// by the vmap(analyze) rule, broadcast across the batch axis) — the handlers
// validate equality, which is also where "stacked distinct tokens" fails
// loudly.
// =============================================================================

#define CUDSS_TOKEN_CHECK(call, msg) \
    do { \
        cudssStatus_t s_ = (call); \
        if (s_ != CUDSS_STATUS_SUCCESS) { \
            return ffi::Error::Internal( \
                std::string("spineax pbatch token: cuDSS call failed (status ") + \
                std::to_string(static_cast<int>(s_)) + "): " msg); \
        } \
    } while (0)

#define CUDA_TOKEN_CHECK(call)                                  \
  do {                                                          \
    cudaError_t err_ = (call);                                  \
    if (err_ != cudaSuccess) {                                  \
      return ffi::Error::Internal(                              \
          std::string("spineax pbatch token: CUDA call failed: ") + \
          cudaGetErrorString(err_));                            \
    }                                                           \
  } while (0)

// Block-diagonal expansion for a BATCHED pattern (B distinct same-shape
// patterns). The uniform-pattern case reuses create_batched_*_kernel above.
__global__ void create_blockdiag_columns_from_batched_kernel(
    const int32_t* __restrict__ batched_in_columns,   // (B, nnz) contiguous
    int32_t* __restrict__ out_columns,                // (B*nnz,)
    int64_t nnz, int64_t n, int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * nnz) {
        int64_t batch_idx = idx / nnz;
        out_columns[idx] = batched_in_columns[idx] + batch_idx * n;
    }
}

__global__ void create_blockdiag_offsets_from_batched_kernel(
    const int32_t* __restrict__ batched_in_offsets,   // (B, n+1) contiguous
    int32_t* __restrict__ out_offsets,                // (B*n + 1,)
    int64_t n, int64_t nnz, int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * n + 1) {
        if (idx == batch_size * n) {
            out_offsets[idx] = batch_size * nnz;
        } else {
            int64_t batch_idx = idx / n;
            int64_t row_idx = idx % n;
            out_offsets[idx] =
                batched_in_offsets[batch_idx * (n + 1) + row_idx] + batch_idx * nnz;
        }
    }
}

// One block-diagonal factorization living in the registry. Not templated:
// dtype is runtime metadata so all dtypes share one registry and id space.
struct BatchFactorEntry {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t   data   = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t x_dummy = nullptr;
    cudssMatrix_t b_dummy = nullptr;
    // device memory OWNED by this entry: the expanded block-diagonal structure
    // and a private copy of the block values
    int32_t* d_offsets = nullptr;  // (B*n + 1,)
    int32_t* d_columns = nullptr;  // (B*nnz,)
    void*    d_values  = nullptr;  // (B*nnz,)
    int64_t block_n = 0, block_nnz = 0, batch = 0;  // system N = batch*block_n
    size_t elem_size = 0;
    cudssDataType_t dtype = CUDSS_R_64F;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    int64_t device_id = 0;
    enum Phase { kAnalyzed = 0, kFactorized = 1 };
    Phase phase = kAnalyzed;
    int32_t ir_nsteps = 0;
    cudaStream_t last_stream = nullptr;
    size_t size_written = 0;

    ~BatchFactorEntry() {
        if (last_stream) cudaStreamSynchronize(last_stream);
        if (A) cudssMatrixDestroy(A);
        if (b_dummy) cudssMatrixDestroy(b_dummy);
        if (x_dummy) cudssMatrixDestroy(x_dummy);
        if (data && handle) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
        if (d_offsets) cudaFree(d_offsets);
        if (d_columns) cudaFree(d_columns);
        if (d_values)  cudaFree(d_values);
    }
};

struct BatchTokenRegistry {
    std::mutex mu;
    std::map<int32_t, std::shared_ptr<BatchFactorEntry>> entries;
    std::list<int32_t> lru;  // front = most recently used
    std::atomic<int32_t> next_id{1};

    static BatchTokenRegistry& instance() {
        static BatchTokenRegistry r;
        return r;
    }
    static size_t capacity() {
        const char* e = std::getenv("SPINEAX_FACTOR_CACHE");
        if (e) { try { return std::stoul(e); } catch (...) {} }
        return 8;
    }
    int32_t insert(std::shared_ptr<BatchFactorEntry> entry) {
        std::lock_guard<std::mutex> lk(mu);
        while (entries.size() >= capacity() && !lru.empty()) {
            int32_t old = lru.back();
            lru.pop_back();
            entries.erase(old);
        }
        int32_t id = next_id.fetch_add(1);
        entries[id] = std::move(entry);
        lru.push_front(id);
        return id;
    }
    std::shared_ptr<BatchFactorEntry> get(int32_t id) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = entries.find(id);
        if (it == entries.end()) return nullptr;
        lru.remove(id);
        lru.push_front(id);
        return it->second;
    }
    bool release(int32_t id) {
        std::lock_guard<std::mutex> lk(mu);
        lru.remove(id);
        return entries.erase(id) > 0;
    }
    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return entries.size();
    }
};

// Read the token buffer (1 or B ids), require all equal, look the entry up.
template <ffi::DataType T>
static ffi::Error batch_token_lookup(cudaStream_t stream,
                                     ffi::Buffer<ffi::S32>& token_buf,
                                     std::shared_ptr<BatchFactorEntry>* out) {
    int64_t count = token_buf.element_count();
    std::vector<int32_t> ids(count);
    CUDA_TOKEN_CHECK(cudaMemcpyAsync(ids.data(), token_buf.typed_data(),
                                     count * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, stream));
    CUDA_TOKEN_CHECK(cudaStreamSynchronize(stream));
    for (int64_t i = 1; i < count; ++i) {
        if (ids[i] != ids[0]) {
            return ffi::Error::Internal(
                "spineax pbatch token: batched token ids differ (" +
                std::to_string(ids[0]) + " vs " + std::to_string(ids[i]) +
                ") — stacked distinct single-system tokens cannot be batch-"
                "operated. Batch them at analysis time (vmap(analyze) or "
                "batch-shaped values) so they share one block-diagonal entry.");
        }
    }
    auto e = BatchTokenRegistry::instance().get(ids[0]);
    if (!e) {
        return ffi::Error::Internal(
            "spineax token: unknown or evicted factorization token " +
            std::to_string(ids[0]) + " (cache capacity " +
            std::to_string(BatchTokenRegistry::capacity()) +
            "; raise with SPINEAX_FACTOR_CACHE, or call release() less eagerly)");
    }
    if (e->dtype != get_cudss_data_type<T>()) {
        return ffi::Error::Internal(
            "spineax pbatch token: dtype mismatch for token " +
            std::to_string(ids[0]));
    }
    if (count != 1 && count != e->batch) {
        return ffi::Error::Internal(
            "spineax pbatch token: got " + std::to_string(count) +
            " token ids for an entry with batch size " +
            std::to_string(e->batch));
    }
    *out = std::move(e);
    return ffi::Error::Success();
}

static ffi::Error batch_token_bind_stream(BatchFactorEntry* e, cudaStream_t stream) {
    if (e->last_stream && e->last_stream != stream) {
        CUDA_TOKEN_CHECK(cudaStreamSynchronize(e->last_stream));
    }
    e->last_stream = stream;
    CUDSS_TOKEN_CHECK(cudssSetStream(e->handle, stream), "cudssSetStream");
    return ffi::Error::Success();
}

static ffi::Error batch_token_set_ir(BatchFactorEntry* e, cudaStream_t stream,
                                     ffi::Buffer<ffi::S32>& ir_buf) {
    int32_t ir = 0;
    CUDA_TOKEN_CHECK(cudaMemcpyAsync(&ir, ir_buf.typed_data(), sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, stream));
    CUDA_TOKEN_CHECK(cudaStreamSynchronize(stream));
    if (ir != e->ir_nsteps) {
        CUDSS_TOKEN_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_IR_N_STEPS,
                                         &ir, sizeof(ir)),
                          "cudssConfigSet ir_nsteps");
        e->ir_nsteps = ir;
    }
    return ffi::Error::Success();
}

// analyze: expand the block-diagonal structure, run block ANALYSIS ===========
template <ffi::DataType T>
static ffi::Error PbatchTokenAnalyze(
    cudaStream_t stream,
    ffi::Buffer<T> csr_values_buf,          // (B*nnz,) contiguous == block values
    ffi::Buffer<ffi::S32> offsets_buf,      // (n+1,) shared or (B*(n+1),) batched
    ffi::Buffer<ffi::S32> columns_buf,      // (nnz,) shared or (B*nnz,) batched
    ffi::ResultBuffer<ffi::S32> token_buf,  // int32[1]
    const int64_t batch_size,
    const int64_t pattern_batched,          // 0: shared pattern, 1: per-block
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id
) {
    using nat = typename get_native_data_type<T>::type;
    CUDA_TOKEN_CHECK(cudaSetDevice(device_id));

    auto e = std::make_shared<BatchFactorEntry>();
    e->last_stream = stream;
    e->batch = batch_size;
    if (pattern_batched) {
        e->block_n = offsets_buf.element_count() / batch_size - 1;
        e->block_nnz = columns_buf.element_count() / batch_size;
    } else {
        e->block_n = offsets_buf.element_count() - 1;
        e->block_nnz = columns_buf.element_count();
    }
    if ((int64_t)csr_values_buf.element_count() != e->batch * e->block_nnz) {
        return ffi::Error::Internal(
            "spineax pbatch token: values size " +
            std::to_string(csr_values_buf.element_count()) + " != batch*nnz = " +
            std::to_string(e->batch * e->block_nnz));
    }
    e->elem_size = sizeof(nat);
    e->dtype = get_cudss_data_type<T>();
    e->device_id = device_id;

    switch (mtype_id) {
        case 0: e->mtype = CUDSS_MTYPE_GENERAL; break;
        case 1: e->mtype = CUDSS_MTYPE_SYMMETRIC; break;
        case 2: e->mtype = CUDSS_MTYPE_HERMITIAN; break;
        case 3: e->mtype = CUDSS_MTYPE_SPD; break;
        case 4: e->mtype = CUDSS_MTYPE_HPD; break;
        default: return ffi::Error::Internal(
            "spineax pbatch token: invalid mtype_id (0 general, 1 symmetric, 2 hermitian, 3 spd, 4 hpd)");
    }
    switch (mview_id) {
        case 0: e->mview = CUDSS_MVIEW_FULL; break;
        case 1: e->mview = CUDSS_MVIEW_UPPER; break;
        case 2: e->mview = CUDSS_MVIEW_LOWER; break;
        default: return ffi::Error::Internal(
            "spineax pbatch token: invalid mview_id (0 full, 1 upper, 2 lower)");
    }

    const int64_t N = e->batch * e->block_n;
    const int64_t NNZ = e->batch * e->block_nnz;

    CUDA_TOKEN_CHECK(cudaMallocAsync(&e->d_offsets, (N + 1) * sizeof(int32_t), stream));
    CUDA_TOKEN_CHECK(cudaMallocAsync(&e->d_columns, NNZ * sizeof(int32_t), stream));
    CUDA_TOKEN_CHECK(cudaMallocAsync(&e->d_values, NNZ * sizeof(nat), stream));

    // Expand the block-diagonal structure into the owned buffers.
    const int threads = 256;
    if (pattern_batched) {
        int64_t total_cols = NNZ;
        create_blockdiag_columns_from_batched_kernel<<<
            (int)((total_cols + threads - 1) / threads), threads, 0, stream>>>(
            columns_buf.typed_data(), e->d_columns, e->block_nnz, e->block_n, e->batch);
        int64_t total_offs = N + 1;
        create_blockdiag_offsets_from_batched_kernel<<<
            (int)((total_offs + threads - 1) / threads), threads, 0, stream>>>(
            offsets_buf.typed_data(), e->d_offsets, e->block_n, e->block_nnz, e->batch);
    } else {
        create_batched_csr_structure(
            offsets_buf.typed_data(), columns_buf.typed_data(),
            e->block_n, e->block_nnz, e->batch,
            &e->d_offsets, &e->d_columns, stream);
    }
    CUDA_TOKEN_CHECK(cudaGetLastError());

    // Private copy of the block values ((B, nnz) contiguous IS the block-diag
    // values array): JAX buffers are transient, later phases re-read A.
    CUDA_TOKEN_CHECK(cudaMemcpyAsync(e->d_values, csr_values_buf.typed_data(),
                                     NNZ * sizeof(nat), cudaMemcpyDeviceToDevice, stream));

    CUDSS_TOKEN_CHECK(cudssCreate(&e->handle), "cudssCreate");
    CUDSS_TOKEN_CHECK(cudssSetStream(e->handle, stream), "cudssSetStream");
    CUDSS_TOKEN_CHECK(cudssConfigCreate(&e->config), "cudssConfigCreate");
    CUDSS_TOKEN_CHECK(cudssDataCreate(e->handle, &e->data), "cudssDataCreate");

    CUDSS_TOKEN_CHECK(cudssMatrixCreateCsr(&e->A, N, N, NNZ,
        e->d_offsets, NULL, e->d_columns, e->d_values,
        CUDSS_R_32I, CUDSS_R_32I, e->dtype,
        e->mtype, e->mview, CUDSS_BASE_ZERO), "cudssMatrixCreateCsr");

    // Placeholder dense descriptors: ANALYSIS/FACTORIZATION never dereference
    // x/b data, but the API requires the objects.
    CUDSS_TOKEN_CHECK(cudssMatrixCreateDn(&e->b_dummy, N, 1, N,
        e->d_values, e->dtype, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn b (dummy)");
    CUDSS_TOKEN_CHECK(cudssMatrixCreateDn(&e->x_dummy, N, 1, N,
        e->d_values, e->dtype, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn x (dummy)");

    CUDSS_TOKEN_CHECK(cudssExecute(e->handle, CUDSS_PHASE_ANALYSIS,
        e->config, e->data, e->A, e->x_dummy, e->b_dummy), "cudssExecute analysis");
    e->phase = BatchFactorEntry::kAnalyzed;

    int32_t id = BatchTokenRegistry::instance().insert(std::move(e));
    CUDA_TOKEN_CHECK(cudaMemcpyAsync(token_buf->typed_data(), &id, sizeof(int32_t),
                                     cudaMemcpyHostToDevice, stream));
    CUDA_TOKEN_CHECK(cudaStreamSynchronize(stream));  // id is a stack local
    return ffi::Error::Success();
}

// factorize / refactorize: block numeric phase ===============================
// Pure phase execution: the only output is the token (dataflow). All post-
// factorization data — diag, permutations, inertia inputs — comes from the
// query handler below, so there is exactly one way to read it.
template <ffi::DataType T, bool kRefactorize>
static ffi::Error PbatchTokenNumeric(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> token_in,         // 1 or B equal ids
    ffi::Buffer<T> csr_values_buf,          // (B*nnz,)
    ffi::Buffer<ffi::S32> ir_buf,           // int32[1]
    ffi::ResultBuffer<ffi::S32> token_out   // same count, same ids
) {
    std::shared_ptr<BatchFactorEntry> e;
    if (auto err = batch_token_lookup<T>(stream, token_in, &e); err.failure()) return err;

    if ((int64_t)csr_values_buf.element_count() != e->batch * e->block_nnz) {
        return ffi::Error::Internal(
            "spineax pbatch token: values size " +
            std::to_string(csr_values_buf.element_count()) + " != batch*nnz = " +
            std::to_string(e->batch * e->block_nnz));
    }
    if (kRefactorize && e->phase < BatchFactorEntry::kFactorized) {
        return ffi::Error::Internal(
            "spineax pbatch token: refactorize requires a factorized token (call factorize first)");
    }

    CUDA_TOKEN_CHECK(cudaSetDevice(e->device_id));
    if (auto err = batch_token_bind_stream(e.get(), stream); err.failure()) return err;
    if (auto err = batch_token_set_ir(e.get(), stream, ir_buf); err.failure()) return err;

    CUDA_TOKEN_CHECK(cudaMemcpyAsync(e->d_values, csr_values_buf.typed_data(),
                                     e->batch * e->block_nnz * e->elem_size,
                                     cudaMemcpyDeviceToDevice, stream));

    CUDSS_TOKEN_CHECK(cudssExecute(e->handle,
        kRefactorize ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION,
        e->config, e->data, e->A, e->x_dummy, e->b_dummy),
        "cudssExecute factorization");
    e->phase = BatchFactorEntry::kFactorized;

    CUDA_TOKEN_CHECK(cudaMemcpyAsync(token_out->typed_data(), token_in.typed_data(),
                                     token_in.element_count() * sizeof(int32_t),
                                     cudaMemcpyDeviceToDevice, stream));
    return ffi::Error::Success();
}

// solve: block SOLVE (multi-RHS via the layout identity on the block system) =
template <ffi::DataType T>
static ffi::Error PbatchTokenSolve(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> token_in,  // 1 or B equal ids
    ffi::Buffer<T> b_values_buf,     // (B*n,) or (R, B*n) row-major
    ffi::Buffer<ffi::S32> ir_buf,    // int32[1]
    ffi::ResultBuffer<T> out_values_buf
) {
    std::shared_ptr<BatchFactorEntry> e;
    if (auto err = batch_token_lookup<T>(stream, token_in, &e); err.failure()) return err;

    if (e->phase < BatchFactorEntry::kFactorized) {
        return ffi::Error::Internal(
            "spineax pbatch token: solve requires a factorized token (call factorize first)");
    }
    const int64_t N = e->batch * e->block_n;
    if (N == 0 || (int64_t)b_values_buf.element_count() % N != 0) {
        return ffi::Error::Internal(
            "spineax pbatch token: rhs size " +
            std::to_string(b_values_buf.element_count()) +
            " is not a multiple of batch*n = " + std::to_string(N));
    }
    int64_t nrhs = b_values_buf.element_count() / N;

    CUDA_TOKEN_CHECK(cudaSetDevice(e->device_id));
    if (auto err = batch_token_bind_stream(e.get(), stream); err.failure()) return err;
    if (auto err = batch_token_set_ir(e.get(), stream, ir_buf); err.failure()) return err;

    cudssMatrix_t bmat = nullptr, xmat = nullptr;
    CUDSS_TOKEN_CHECK(cudssMatrixCreateDn(&bmat, N, nrhs, N,
        const_cast<typename get_native_data_type<T>::type*>(b_values_buf.typed_data()),
        e->dtype, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn b (solve)");
    CUDSS_TOKEN_CHECK(cudssMatrixCreateDn(&xmat, N, nrhs, N,
        out_values_buf->typed_data(), e->dtype, CUDSS_LAYOUT_COL_MAJOR),
        "cudssMatrixCreateDn x (solve)");

    cudssStatus_t solve_status = cudssExecute(e->handle, CUDSS_PHASE_SOLVE,
        e->config, e->data, e->A, xmat, bmat);
    cudssMatrixDestroy(bmat);
    cudssMatrixDestroy(xmat);
    if (solve_status != CUDSS_STATUS_SUCCESS) {
        return ffi::Error::Internal(
            "spineax pbatch token: cuDSS solve failed (status " +
            std::to_string(static_cast<int>(solve_status)) + ")");
    }
    return ffi::Error::Success();
}

// query: read every cuDSS data item from a factorized token =================
// Subsumes single_solve_re.cpp: everything is returned unconditionally
// (zero-filled where cuDSS declines for this matrix type / config) and Python
// figures out what it wants. Array outputs are sized by the block system
// (N = batch * block_n); scalar outputs are block-global.
static constexpr int64_t kNdPartitionTreeSize = (1 << 10) - 1;  // nd_nlevels=10 default

template <ffi::DataType T>
static ffi::Error PbatchTokenQuery(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> token_in,                    // 1 or B equal ids
    ffi::ResultBuffer<ffi::S64> lu_nnz_buf,            // [1]
    ffi::ResultBuffer<ffi::S32> npivots_buf,           // [1]
    ffi::ResultBuffer<ffi::S32> inertia_buf,           // [2] cuDSS native (block-global)
    ffi::ResultBuffer<ffi::S32> perm_reorder_row_buf,  // [N]
    ffi::ResultBuffer<ffi::S32> perm_reorder_col_buf,  // [N]
    ffi::ResultBuffer<ffi::S32> perm_row_buf,          // [N] (reordering alg 1/2 only)
    ffi::ResultBuffer<ffi::S32> perm_col_buf,          // [N] (reordering alg 1/2 only)
    ffi::ResultBuffer<ffi::S32> perm_matching_buf,     // [N]
    ffi::ResultBuffer<T> diag_buf,                     // [N]
    ffi::ResultBuffer<ffi::F32> scale_row_buf,         // [N]
    ffi::ResultBuffer<ffi::F32> scale_col_buf,         // [N]
    ffi::ResultBuffer<ffi::S32> nd_partition_tree_buf, // [kNdPartitionTreeSize]
    ffi::ResultBuffer<ffi::S32> nsuperpanels_buf,      // [1]
    ffi::ResultBuffer<ffi::S64> schur_shape_buf        // [2]
) {
    std::shared_ptr<BatchFactorEntry> e;
    if (auto err = batch_token_lookup<T>(stream, token_in, &e); err.failure()) return err;
    if (e->phase < BatchFactorEntry::kFactorized) {
        return ffi::Error::Internal(
            "spineax token: query requires a factorized token (call factorize first)");
    }
    CUDA_TOKEN_CHECK(cudaSetDevice(e->device_id));
    if (auto err = batch_token_bind_stream(e.get(), stream); err.failure()) return err;

    const int64_t N = e->batch * e->block_n;
    // The output buffers are sized by the caller's static token metadata; a
    // mismatch (e.g. query of a vmap-minted batch token from inside vmap)
    // must fail loudly rather than overrun the buffers.
    if ((int64_t)diag_buf->element_count() != N) {
        return ffi::Error::Internal(
            "spineax token: query output size " +
            std::to_string(diag_buf->element_count()) +
            " != block system dimension " + std::to_string(N) +
            " (query is an eager/outer-level operation — call it outside "
            "vmap with batch-shaped token metadata)");
    }
    size_t written = 0;

    // host-side scalars: dataGet to host, then H2D into the result buffer;
    // zero on failure so Python always gets well-defined values
    #define QUERY_HOST_SCALAR(PARAM, TYPE, COUNT, BUF) \
        do { \
            TYPE tmp_[COUNT] = {}; \
            if (cudssDataGet(e->handle, e->data, PARAM, tmp_, sizeof(tmp_), \
                             &written) != CUDSS_STATUS_SUCCESS) { \
                for (int i_ = 0; i_ < (COUNT); ++i_) tmp_[i_] = 0; \
            } \
            CUDA_TOKEN_CHECK(cudaMemcpy((BUF)->typed_data(), tmp_, sizeof(tmp_), \
                                        cudaMemcpyHostToDevice)); \
        } while (0)

    // device-side arrays: dataGet writes the device buffer directly
    #define QUERY_DEVICE_ARRAY(PARAM, BUF, BYTES) \
        do { \
            if (cudssDataGet(e->handle, e->data, PARAM, (BUF)->typed_data(), \
                             (BYTES), &written) != CUDSS_STATUS_SUCCESS) { \
                CUDA_TOKEN_CHECK(cudaMemset((BUF)->typed_data(), 0, (BYTES))); \
            } \
        } while (0)

    QUERY_HOST_SCALAR(CUDSS_DATA_LU_NNZ, int64_t, 1, lu_nnz_buf);
    QUERY_HOST_SCALAR(CUDSS_DATA_NPIVOTS, int32_t, 1, npivots_buf);
    QUERY_HOST_SCALAR(CUDSS_DATA_INERTIA, int32_t, 2, inertia_buf);
    QUERY_HOST_SCALAR(CUDSS_DATA_NSUPERPANELS, int32_t, 1, nsuperpanels_buf);
    QUERY_HOST_SCALAR(CUDSS_DATA_SCHUR_SHAPE, int64_t, 2, schur_shape_buf);

    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_REORDER_ROW, perm_reorder_row_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_REORDER_COL, perm_reorder_col_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_ROW, perm_row_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_COL, perm_col_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_MATCHING, perm_matching_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_DIAG, diag_buf, N * (int64_t)e->elem_size);
    QUERY_DEVICE_ARRAY(CUDSS_DATA_SCALE_ROW, scale_row_buf, N * sizeof(float));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_SCALE_COL, scale_col_buf, N * sizeof(float));
    // cuDSS >= 0.8 removed CUDSS_DATA_ELIMINATION_TREE; the nested-dissection
    // partition tree is its successor and exposes the same reordering structure.
    QUERY_DEVICE_ARRAY(CUDSS_DATA_ND_PARTITION_TREE, nd_partition_tree_buf,
                       kNdPartitionTreeSize * sizeof(int32_t));

    #undef QUERY_HOST_SCALAR
    #undef QUERY_DEVICE_ARRAY
    return ffi::Error::Success();
}

// token FFI handler definitions ===============================================
#define DEFINE_PBATCH_TOKEN_FFI_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenAnalyze##TypeName, PbatchTokenAnalyze<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<int64_t>("batch_size") \
            .Attr<int64_t>("pattern_batched") \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenFactorize##TypeName, (PbatchTokenNumeric<DataType, false>), \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>()); \
    \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenRefactorize##TypeName, (PbatchTokenNumeric<DataType, true>), \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>()); \
    \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenSolve##TypeName, PbatchTokenSolve<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>()); \
    \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenQuery##TypeName, PbatchTokenQuery<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S64>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::F32>>() \
            .Ret<ffi::Buffer<ffi::F32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S64>>());

DEFINE_PBATCH_TOKEN_FFI_HANDLERS(f32, ffi::F32);
DEFINE_PBATCH_TOKEN_FFI_HANDLERS(f64, ffi::F64);
DEFINE_PBATCH_TOKEN_FFI_HANDLERS(c64, ffi::C64);
DEFINE_PBATCH_TOKEN_FFI_HANDLERS(c128, ffi::C128);

#define EXPORT_PBATCH_TOKEN_HANDLERS(m, TypeName) \
    m.def("token_handlers_" #TypeName, []() { \
        nb::dict d; \
        d["analyze"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenAnalyze##TypeName)); \
        d["factorize"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenFactorize##TypeName)); \
        d["refactorize"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenRefactorize##TypeName)); \
        d["solve"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenSolve##TypeName)); \
        d["query"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenQuery##TypeName)); \
        return d; \
    });

// generate all nanobind modules! :)
NB_MODULE(pbatch_solve, m) {

    EXPORT_PBATCH_TOKEN_HANDLERS(m, f32);
    EXPORT_PBATCH_TOKEN_HANDLERS(m, f64);
    EXPORT_PBATCH_TOKEN_HANDLERS(m, c64);
    EXPORT_PBATCH_TOKEN_HANDLERS(m, c128);

    m.def("token_release", [](int32_t id) {
        return BatchTokenRegistry::instance().release(id);
    });
    m.def("token_registry_size", []() {
        return BatchTokenRegistry::instance().size();
    });
    m.def("token_cache_capacity", []() {
        return BatchTokenRegistry::instance().capacity();
    });
    m.def("nd_partition_tree_size", []() {
        return kNdPartitionTreeSize;
    });
}
