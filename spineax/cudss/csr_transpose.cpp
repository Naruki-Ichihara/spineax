#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "cuda_runtime_api.h"
#include "cusparse.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

template <ffi::DataType T>
cudaDataType_t cuda_data_type();

template <>
cudaDataType_t cuda_data_type<ffi::F32>() {
    return CUDA_R_32F;
}

template <>
cudaDataType_t cuda_data_type<ffi::F64>() {
    return CUDA_R_64F;
}

template <>
cudaDataType_t cuda_data_type<ffi::C64>() {
    return CUDA_C_32F;
}

template <>
cudaDataType_t cuda_data_type<ffi::C128>() {
    return CUDA_C_64F;
}

static ffi::Error cusparse_error(cusparseStatus_t status, const char* operation) {
    return ffi::Error::Internal(
        std::string("Spineax cuSPARSE ") + operation + " failed: " +
        cusparseGetErrorString(status));
}

static ffi::Error cuda_error(cudaError_t status, const char* operation) {
    return ffi::Error::Internal(
        std::string("Spineax CUDA ") + operation + " failed: " +
        cudaGetErrorString(status));
}

struct CuSparseHandle {
    std::mutex mutex;
    cusparseHandle_t value = nullptr;
};

static CuSparseHandle& handle_for_stream(int device, cudaStream_t stream) {
    using HandleKey = std::pair<int, std::uintptr_t>;
    // Cache one handle per observed device and stream. The map intentionally
    // survives static destruction, when the CUDA runtime may already be gone.
    static auto* mutex = new std::mutex;
    static auto* handles =
        new std::map<HandleKey, std::unique_ptr<CuSparseHandle>>;
    std::lock_guard lock(*mutex);
    HandleKey key{device, reinterpret_cast<std::uintptr_t>(stream)};
    auto& handle = (*handles)[key];
    if (!handle) {
        handle = std::make_unique<CuSparseHandle>();
    }
    return *handle;
}

static cusparseStatus_t acquire_handle(
    int device,
    cudaStream_t stream,
    std::unique_lock<std::mutex>& lock,
    cusparseHandle_t& handle
) {
    CuSparseHandle& state = handle_for_stream(device, stream);
    lock = std::unique_lock(state.mutex);
    if (state.value == nullptr) {
        cusparseStatus_t status = cusparseCreate(&state.value);
        if (status != CUSPARSE_STATUS_SUCCESS) return status;
        status = cusparseSetStream(state.value, stream);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cusparseDestroy(state.value);
            state.value = nullptr;
            return status;
        }
    }
    handle = state.value;
    return CUSPARSE_STATUS_SUCCESS;
}

static ffi::Error CsrTransposeOrder(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> offsets,
    ffi::Buffer<ffi::S32> columns,
    ffi::Buffer<ffi::S32> identity,
    ffi::ResultBuffer<ffi::S32> order
) {
    const int64_t n64 = offsets.element_count() - 1;
    const int64_t nnz64 = columns.element_count();
    if (n64 < 1 || n64 > std::numeric_limits<int>::max() ||
        nnz64 > std::numeric_limits<int>::max()) {
        return ffi::Error::InvalidArgument(
            "Spineax cuSPARSE transpose requires positive int32 dimensions");
    }
    if (identity.element_count() != columns.element_count() ||
        order->element_count() != columns.element_count()) {
        return ffi::Error::InvalidArgument(
            "Spineax cuSPARSE transpose order received inconsistent buffer sizes");
    }

    const int n = static_cast<int>(n64);
    const int nnz = static_cast<int>(nnz64);
    if (nnz == 0) return ffi::Error::Success();
    int device = -1;
    cudaError_t cuda_status = cudaGetDevice(&device);
    if (cuda_status != cudaSuccess) {
        return cuda_error(cuda_status, "device query");
    }
    std::unique_lock<std::mutex> handle_lock;
    cusparseHandle_t handle = nullptr;
    cusparseStatus_t status = acquire_handle(
        device, stream, handle_lock, handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_error(status, "handle creation");
    }

    size_t sort_size = 0;
    status = cusparseXcoosort_bufferSizeExt(
        handle, n, n, nnz, order->typed_data(),
        columns.typed_data(), &sort_size);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_error(status, "sort workspace query");
    }

    const size_t columns_size = static_cast<size_t>(nnz) * sizeof(int32_t);
    const size_t columns_offset = (columns_size + 255) & ~size_t{255};
    const size_t sort_offset =
        (columns_offset + columns_size + 255) & ~size_t{255};
    void* workspace = nullptr;
    cuda_status = cudaMallocAsync(
        &workspace, sort_offset + sort_size, stream);
    if (cuda_status != cudaSuccess) {
        return cuda_error(cuda_status, "order workspace allocation");
    }
    auto* rows = static_cast<int32_t*>(workspace);
    auto* sorted_columns = reinterpret_cast<int32_t*>(
        static_cast<unsigned char*>(workspace) + columns_offset);
    auto* sort_workspace = static_cast<unsigned char*>(workspace) + sort_offset;
    cuda_status = cudaMemcpyAsync(
        sorted_columns, columns.typed_data(), columns_size,
        cudaMemcpyDeviceToDevice, stream);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpyAsync(
            order->typed_data(), identity.typed_data(), columns_size,
            cudaMemcpyDeviceToDevice, stream);
    }
    if (cuda_status != cudaSuccess) {
        cudaFreeAsync(workspace, stream);
        return cuda_error(cuda_status, "order initialization");
    }

    status = cusparseXcsr2coo(
        handle, offsets.typed_data(), nnz, n,
        rows, CUSPARSE_INDEX_BASE_ZERO);
    if (status == CUSPARSE_STATUS_SUCCESS) {
        status = cusparseXcoosortByColumn(
            handle, n, n, nnz, rows,
            sorted_columns, order->typed_data(), sort_workspace);
    }

    cuda_status = cudaFreeAsync(workspace, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_error(status, "transpose order preparation");
    }
    if (cuda_status != cudaSuccess) {
        return cuda_error(cuda_status, "order workspace release");
    }
    return ffi::Error::Success();
}

template <ffi::DataType T>
static ffi::Error CsrTranspose(
    cudaStream_t stream,
    ffi::Buffer<T> values,
    ffi::Buffer<ffi::S32> offsets,
    ffi::Buffer<ffi::S32> columns,
    ffi::ResultBuffer<T> transposed_values,
    ffi::ResultBuffer<ffi::S32> transposed_offsets,
    ffi::ResultBuffer<ffi::S32> transposed_columns
) {
    const int64_t n64 = offsets.element_count() - 1;
    const int64_t nnz64 = columns.element_count();
    if (n64 < 1 || n64 > std::numeric_limits<int>::max() ||
        nnz64 > std::numeric_limits<int>::max()) {
        return ffi::Error::InvalidArgument(
            "Spineax cuSPARSE transpose requires positive int32 dimensions");
    }
    if (values.element_count() != columns.element_count() ||
        transposed_values->element_count() != columns.element_count() ||
        transposed_offsets->element_count() != offsets.element_count() ||
        transposed_columns->element_count() != columns.element_count()) {
        return ffi::Error::InvalidArgument(
            "Spineax cuSPARSE transpose received inconsistent buffer sizes");
    }

    const int n = static_cast<int>(n64);
    const int nnz = static_cast<int>(nnz64);
    int device = -1;
    cudaError_t cuda_status = cudaGetDevice(&device);
    if (cuda_status != cudaSuccess) {
        return cuda_error(cuda_status, "device query");
    }
    std::unique_lock<std::mutex> handle_lock;
    cusparseHandle_t handle = nullptr;
    cusparseStatus_t status = acquire_handle(
        device, stream, handle_lock, handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_error(status, "handle creation");
    }

    size_t workspace_size = 0;
    status = cusparseCsr2cscEx2_bufferSize(
        handle, n, n, nnz, values.typed_data(), offsets.typed_data(),
        columns.typed_data(), transposed_values->typed_data(),
        transposed_offsets->typed_data(), transposed_columns->typed_data(),
        cuda_data_type<T>(), CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT, &workspace_size);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_error(status, "workspace query");
    }

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cuda_status = cudaMallocAsync(&workspace, workspace_size, stream);
        if (cuda_status != cudaSuccess) {
            return cuda_error(cuda_status, "workspace allocation");
        }
    }

    status = cusparseCsr2cscEx2(
        handle, n, n, nnz, values.typed_data(), offsets.typed_data(),
        columns.typed_data(), transposed_values->typed_data(),
        transposed_offsets->typed_data(), transposed_columns->typed_data(),
        cuda_data_type<T>(), CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT, workspace);

    if (workspace != nullptr) {
        cuda_status = cudaFreeAsync(workspace, stream);
        if (status == CUSPARSE_STATUS_SUCCESS && cuda_status != cudaSuccess) {
            return cuda_error(cuda_status, "workspace release");
        }
    }
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_error(status, "CSR transpose");
    }
    return ffi::Error::Success();
}

#define DEFINE_HANDLER(Name, Type)                                      \
    XLA_FFI_DEFINE_HANDLER(                                             \
        kCsrTranspose##Name, CsrTranspose<Type>,                        \
        ffi::Ffi::Bind()                                                \
            .Ctx<ffi::PlatformStream<cudaStream_t>>()                   \
            .Arg<ffi::Buffer<Type>>()                                   \
            .Arg<ffi::Buffer<ffi::S32>>()                               \
            .Arg<ffi::Buffer<ffi::S32>>()                               \
            .Ret<ffi::Buffer<Type>>()                                   \
            .Ret<ffi::Buffer<ffi::S32>>()                               \
            .Ret<ffi::Buffer<ffi::S32>>());

DEFINE_HANDLER(f32, ffi::F32);
DEFINE_HANDLER(f64, ffi::F64);
DEFINE_HANDLER(c64, ffi::C64);
DEFINE_HANDLER(c128, ffi::C128);

XLA_FFI_DEFINE_HANDLER(
    kCsrTransposeOrder, CsrTransposeOrder,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

#define EXPORT_HANDLER(module, Name)                                    \
    module.def("handler_" #Name, []() {                                \
        return nb::capsule(reinterpret_cast<void*>(kCsrTranspose##Name)); \
    });

void register_csr_transpose_handlers(nb::module_& module) {
    EXPORT_HANDLER(module, f32);
    EXPORT_HANDLER(module, f64);
    EXPORT_HANDLER(module, c64);
    EXPORT_HANDLER(module, c128);
    module.def("order_handler", []() {
        return nb::capsule(reinterpret_cast<void*>(kCsrTransposeOrder));
    });
}
