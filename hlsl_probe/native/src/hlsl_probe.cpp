#include "hlsl_probe_c_api.h"
#include "hlsl_probe_config.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

std::string g_last_error;

std::string hresult_hex(HRESULT hr) {
    std::ostringstream oss;
    oss << "0x" << std::uppercase << std::hex << std::setw(8) << std::setfill('0')
        << static_cast<unsigned long>(static_cast<uint32_t>(hr));
    return oss.str();
}

std::string wide_to_utf8(const wchar_t* value) {
    if (!value) {
        return {};
    }
    const int required = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 0) {
        return {};
    }
    std::string output(static_cast<size_t>(required - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, output.data(), required, nullptr, nullptr);
    return output;
}

std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 16);
    for (char ch : value) {
        switch (ch) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                std::ostringstream oss;
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(static_cast<unsigned char>(ch));
                output += oss.str();
            } else {
                output += ch;
            }
            break;
        }
    }
    return output;
}

void throw_if_failed(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        std::ostringstream oss;
        oss << what << " failed with HRESULT " << hresult_hex(hr);
        throw std::runtime_error(oss.str());
    }
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

std::filesystem::path module_directory() {
    HMODULE module = nullptr;
    const auto flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&module_directory), &module)) {
        throw std::runtime_error("GetModuleHandleExW failed");
    }

    std::wstring buffer(MAX_PATH, L'\0');
    while (true) {
        DWORD length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            throw std::runtime_error("GetModuleFileNameW failed");
        }
        if (length < buffer.size() - 1) {
            buffer.resize(length);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    return std::filesystem::path(buffer).parent_path();
}

std::string default_sdk_path_utf8() {
    auto path = module_directory() / "D3D12";
    auto text = path.string();
    if (!text.empty() && text.back() != '\\' && text.back() != '/') {
        text.push_back('\\');
    }
    return text;
}

char* copy_to_c_string(const std::string& value) {
    auto* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

const char* wave_mma_tier_name(D3D12_WAVE_MMA_TIER tier) {
    switch (tier) {
    case D3D12_WAVE_MMA_TIER_NOT_SUPPORTED: return "not_supported";
    case D3D12_WAVE_MMA_TIER_1_0: return "1_0";
    default: return "unknown";
    }
}

const char* shader_model_name(D3D_SHADER_MODEL shader_model) {
    switch (shader_model) {
    case D3D_SHADER_MODEL_6_10: return "6.10";
    case D3D_SHADER_MODEL_6_9: return "6.9";
    case D3D_SHADER_MODEL_6_8: return "6.8";
    case D3D_SHADER_MODEL_6_7: return "6.7";
    case D3D_SHADER_MODEL_6_6: return "6.6";
    case D3D_SHADER_MODEL_6_5: return "6.5";
    case D3D_SHADER_MODEL_6_4: return "6.4";
    case D3D_SHADER_MODEL_6_3: return "6.3";
    case D3D_SHADER_MODEL_6_2: return "6.2";
    case D3D_SHADER_MODEL_6_1: return "6.1";
    case D3D_SHADER_MODEL_6_0: return "6.0";
    default: return "unknown";
    }
}

const char* linear_algebra_tier_name(D3D12_LINEAR_ALGEBRA_TIER tier) {
    switch (tier) {
    case D3D12_LINEAR_ALGEBRA_TIER_NOT_SUPPORTED: return "not_supported";
    case D3D12_LINEAR_ALGEBRA_TIER_1_0: return "1_0";
    default: return "unknown";
    }
}

const char* linear_algebra_datatype_name(D3D12_LINEAR_ALGEBRA_DATATYPE datatype) {
    switch (datatype) {
    case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT16: return "sint16";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT16: return "uint16";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32: return "sint32";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT32: return "uint32";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16: return "float16";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32: return "float32";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8: return "sint8";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8: return "uint8";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E4M3FN: return "float8_e4m3fn";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E5M2: return "float8_e5m2";
    default: return "unknown";
    }
}

std::string linear_algebra_support_flags_json(D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAGS flags) {
    std::ostringstream oss;
    oss << "{\"value\":" << static_cast<uint32_t>(flags)
        << ",\"supported\":" << ((flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED) ? "true" : "false")
        << ",\"emulated_inputs\":" << ((flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_INPUTS) ? "true" : "false")
        << ",\"emulated_outputs\":" << ((flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_OUTPUTS) ? "true" : "false")
        << ",\"transpose\":" << ((flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_TRANSPOSE) ? "true" : "false")
        << "}";
    return oss.str();
}

struct GpuBuffer {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> readback;
    HlslProbeBufferDesc desc = {};
    uint64_t requested_size = 0;
    uint64_t resource_size = 0;
};

struct AdapterInfo {
    std::string name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint64_t dedicated_video_memory = 0;
};

const char* view_kind_name(uint32_t view_kind) {
    switch (view_kind) {
    case HLSL_PROBE_BUFFER_VIEW_RAW: return "raw";
    case HLSL_PROBE_BUFFER_VIEW_TYPED: return "typed";
    default: return "unknown";
    }
}

const char* buffer_format_name(uint32_t format) {
    switch (format) {
    case HLSL_PROBE_BUFFER_FORMAT_RAW_U32: return "raw_u32";
    case HLSL_PROBE_BUFFER_FORMAT_F16: return "float16";
    case HLSL_PROBE_BUFFER_FORMAT_F32: return "float32";
    case HLSL_PROBE_BUFFER_FORMAT_I8: return "int8";
    case HLSL_PROBE_BUFFER_FORMAT_U8: return "uint8";
    case HLSL_PROBE_BUFFER_FORMAT_I32: return "int32";
    case HLSL_PROBE_BUFFER_FORMAT_U32: return "uint32";
    default: return "unknown";
    }
}

uint64_t bytes_per_element(uint32_t format) {
    switch (format) {
    case HLSL_PROBE_BUFFER_FORMAT_RAW_U32: return 4;
    case HLSL_PROBE_BUFFER_FORMAT_F16: return 2;
    case HLSL_PROBE_BUFFER_FORMAT_F32: return 4;
    case HLSL_PROBE_BUFFER_FORMAT_I8: return 1;
    case HLSL_PROBE_BUFFER_FORMAT_U8: return 1;
    case HLSL_PROBE_BUFFER_FORMAT_I32: return 4;
    case HLSL_PROBE_BUFFER_FORMAT_U32: return 4;
    default: return 0;
    }
}

DXGI_FORMAT dxgi_format_for(uint32_t format) {
    switch (format) {
    case HLSL_PROBE_BUFFER_FORMAT_RAW_U32: return DXGI_FORMAT_R32_TYPELESS;
    case HLSL_PROBE_BUFFER_FORMAT_F16: return DXGI_FORMAT_R16_FLOAT;
    case HLSL_PROBE_BUFFER_FORMAT_F32: return DXGI_FORMAT_R32_FLOAT;
    case HLSL_PROBE_BUFFER_FORMAT_I8: return DXGI_FORMAT_R8_SINT;
    case HLSL_PROBE_BUFFER_FORMAT_U8: return DXGI_FORMAT_R8_UINT;
    case HLSL_PROBE_BUFFER_FORMAT_I32: return DXGI_FORMAT_R32_SINT;
    case HLSL_PROBE_BUFFER_FORMAT_U32: return DXGI_FORMAT_R32_UINT;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

std::string buffer_desc_json(const HlslProbeBufferDesc& desc) {
    std::ostringstream oss;
    oss << "{\"view_kind\":\"" << view_kind_name(desc.view_kind)
        << "\",\"format\":\"" << buffer_format_name(desc.format)
        << "\",\"element_count\":" << desc.element_count
        << ",\"size_bytes\":" << desc.size_bytes << "}";
    return oss.str();
}

} // namespace

struct HlslProbeContext {
    std::string last_error;
    std::string sdk_path;
    uint32_t sdk_version_used = 0;
    HRESULT global_experimental_shader_models_hr = E_NOTIMPL;
    HRESULT factory_experimental_shader_models_hr = E_NOTIMPL;
    AdapterInfo adapter_info;
    ComPtr<IDXGIFactory6> dxgi_factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12DeviceFactory> device_factory;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;

    explicit HlslProbeContext(std::string sdk_path_in) : sdk_path(std::move(sdk_path_in)) {
        initialize();
    }

    ~HlslProbeContext() {
        if (fence_event) {
            CloseHandle(fence_event);
        }
    }

    void initialize() {
        if (sdk_path.empty()) {
            sdk_path = default_sdk_path_utf8();
        }
        if (!std::filesystem::exists(std::filesystem::path(sdk_path) / "D3D12Core.dll")) {
            std::ostringstream oss;
            oss << "Agility SDK runtime not found at " << sdk_path << "D3D12Core.dll";
            throw std::runtime_error(oss.str());
        }

        UINT dxgi_flags = 0;
#if defined(_DEBUG)
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            dxgi_flags |= DXGI_CREATE_FACTORY_DEBUG;
        }
#endif

        throw_if_failed(CreateDXGIFactory2(dxgi_flags, IID_PPV_ARGS(&dxgi_factory)), "CreateDXGIFactory2");

        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            HRESULT hr = dxgi_factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            throw_if_failed(hr, "EnumAdapterByGpuPreference");

            DXGI_ADAPTER_DESC1 desc = {};
            candidate->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                continue;
            }
            adapter = candidate;
            adapter_info.name = wide_to_utf8(desc.Description);
            adapter_info.vendor_id = desc.VendorId;
            adapter_info.device_id = desc.DeviceId;
            adapter_info.dedicated_video_memory = desc.DedicatedVideoMemory;
            break;
        }

        if (!adapter) {
            throw std::runtime_error("No hardware DXGI adapter found");
        }

        const IID experimental_features[] = { D3D12ExperimentalShaderModels };
        global_experimental_shader_models_hr = D3D12EnableExperimentalFeatures(1, experimental_features, nullptr, nullptr);

        ComPtr<ID3D12SDKConfiguration1> sdk_config;
        throw_if_failed(D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&sdk_config)), "D3D12GetInterface(CLSID_D3D12SDKConfiguration)");

        HRESULT factory_hr = sdk_config->CreateDeviceFactory(HLSL_PROBE_AGILITY_SDK_VERSION, sdk_path.c_str(), IID_PPV_ARGS(&device_factory));
        sdk_version_used = HLSL_PROBE_AGILITY_SDK_VERSION;

#if defined(D3D12_SDK_VERSION)
        HRESULT fallback_factory_hr = S_OK;
    const bool has_distinct_fallback_sdk = HLSL_PROBE_AGILITY_SDK_VERSION != D3D12_SDK_VERSION;
    if (FAILED(factory_hr) && has_distinct_fallback_sdk) {
            fallback_factory_hr = sdk_config->CreateDeviceFactory(D3D12_SDK_VERSION, sdk_path.c_str(), IID_PPV_ARGS(&device_factory));
            if (SUCCEEDED(fallback_factory_hr)) {
                factory_hr = fallback_factory_hr;
                sdk_version_used = D3D12_SDK_VERSION;
            }
        }
#endif

        if (FAILED(factory_hr)) {
            std::ostringstream oss;
            oss << "ID3D12SDKConfiguration1::CreateDeviceFactory failed with HRESULT " << hresult_hex(factory_hr)
                << " for SDK version " << HLSL_PROBE_AGILITY_SDK_VERSION
                << "; D3D12EnableExperimentalFeatures(D3D12ExperimentalShaderModels) returned " << hresult_hex(global_experimental_shader_models_hr);
#if defined(D3D12_SDK_VERSION)
            if (has_distinct_fallback_sdk) {
                oss << "; fallback SDK version " << D3D12_SDK_VERSION << " returned " << hresult_hex(fallback_factory_hr);
            }
#endif
            throw std::runtime_error(oss.str());
        }

        factory_experimental_shader_models_hr = device_factory->EnableExperimentalFeatures(1, experimental_features, nullptr, nullptr);
        throw_if_failed(device_factory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "ID3D12DeviceFactory::CreateDevice");

        D3D12_COMMAND_QUEUE_DESC queue_desc = {};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        throw_if_failed(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        throw_if_failed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
        throw_if_failed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&command_list)), "CreateCommandList");
        command_list->Close();

        throw_if_failed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event) {
            throw std::runtime_error("CreateEventW failed");
        }
    }

    void wait_for_gpu() {
        const uint64_t value = ++fence_value;
        throw_if_failed(queue->Signal(fence.Get(), value), "ID3D12CommandQueue::Signal");
        if (fence->GetCompletedValue() < value) {
            throw_if_failed(fence->SetEventOnCompletion(value, fence_event), "ID3D12Fence::SetEventOnCompletion");
            WaitForSingleObject(fence_event, INFINITE);
        }
    }

    ComPtr<ID3D12Resource> create_buffer(uint64_t size, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES initial_state, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        D3D12_HEAP_PROPERTIES heap_props = {};
        heap_props.Type = heap_type;

        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = std::max<uint64_t>(size, 4);
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;

        ComPtr<ID3D12Resource> resource;
        throw_if_failed(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc, initial_state, nullptr, IID_PPV_ARGS(&resource)), "CreateCommittedResource(buffer)");
        return resource;
    }

    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        command_list->ResourceBarrier(1, &barrier);
    }

    void validate_format_support(const HlslProbeBufferDesc& desc, bool output, const std::string& label) {
        const DXGI_FORMAT dxgi_format = dxgi_format_for(desc.format);
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support = {};
        support.Format = dxgi_format;
        HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support));
        if (FAILED(hr)) {
            std::ostringstream oss;
            oss << label << " CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT) failed for "
                << buffer_format_name(desc.format) << " with HRESULT " << hresult_hex(hr);
            throw std::runtime_error(oss.str());
        }

        if ((support.Support1 & D3D12_FORMAT_SUPPORT1_BUFFER) == 0) {
            std::ostringstream oss;
            oss << label << " format " << buffer_format_name(desc.format) << " is not supported as a buffer view";
            throw std::runtime_error(oss.str());
        }

        if (!output && (support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD) == 0) {
            std::ostringstream oss;
            oss << label << " format " << buffer_format_name(desc.format) << " is not supported for shader loads";
            throw std::runtime_error(oss.str());
        }

        if (output) {
            const bool has_uav_view = (support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0;
            const bool has_uav_store = (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;
            if (!has_uav_view || !has_uav_store) {
                std::ostringstream oss;
                oss << label << " format " << buffer_format_name(desc.format)
                    << " is not supported for typed UAV stores"
                    << " (Support1=0x" << std::hex << support.Support1
                    << ", Support2=0x" << support.Support2 << ")";
                throw std::runtime_error(oss.str());
            }
        }
    }

    void validate_buffer_desc(const HlslProbeBufferDesc& desc, bool output, uint32_t index) {
        std::ostringstream label_oss;
        label_oss << (output ? "output" : "input") << "[" << index << "]";
        const std::string label = label_oss.str();

        if (desc.struct_size != sizeof(HlslProbeBufferDesc)) {
            std::ostringstream oss;
            oss << label << " has invalid HlslProbeBufferDesc.struct_size " << desc.struct_size
                << "; expected " << sizeof(HlslProbeBufferDesc);
            throw std::runtime_error(oss.str());
        }
        if (desc.element_count > static_cast<uint64_t>(std::numeric_limits<UINT>::max())) {
            std::ostringstream oss;
            oss << label << " element_count exceeds D3D12 descriptor UINT range: " << desc.element_count;
            throw std::runtime_error(oss.str());
        }

        if (desc.view_kind == HLSL_PROBE_BUFFER_VIEW_RAW) {
            if (desc.format != HLSL_PROBE_BUFFER_FORMAT_RAW_U32) {
                std::ostringstream oss;
                oss << label << " raw view requires raw_u32 format, got " << buffer_format_name(desc.format);
                throw std::runtime_error(oss.str());
            }
            if (desc.size_bytes % 4 != 0) {
                std::ostringstream oss;
                oss << label << " raw view size_bytes must be 4-byte aligned, got " << desc.size_bytes;
                throw std::runtime_error(oss.str());
            }
            if (desc.element_count != desc.size_bytes / 4) {
                std::ostringstream oss;
                oss << label << " raw view element_count must equal size_bytes / 4; got element_count="
                    << desc.element_count << ", size_bytes=" << desc.size_bytes;
                throw std::runtime_error(oss.str());
            }
            return;
        }

        if (desc.view_kind != HLSL_PROBE_BUFFER_VIEW_TYPED) {
            std::ostringstream oss;
            oss << label << " has unknown view_kind " << desc.view_kind;
            throw std::runtime_error(oss.str());
        }
        if (desc.format == HLSL_PROBE_BUFFER_FORMAT_RAW_U32) {
            std::ostringstream oss;
            oss << label << " typed view cannot use raw_u32 format";
            throw std::runtime_error(oss.str());
        }
        const uint64_t bytes = bytes_per_element(desc.format);
        if (bytes == 0 || dxgi_format_for(desc.format) == DXGI_FORMAT_UNKNOWN) {
            std::ostringstream oss;
            oss << label << " has unknown buffer format " << desc.format;
            throw std::runtime_error(oss.str());
        }
        if (desc.element_count > std::numeric_limits<uint64_t>::max() / bytes) {
            std::ostringstream oss;
            oss << label << " size calculation overflow for format " << buffer_format_name(desc.format);
            throw std::runtime_error(oss.str());
        }
        const uint64_t expected_size = desc.element_count * bytes;
        if (desc.size_bytes != expected_size) {
            std::ostringstream oss;
            oss << label << " typed " << buffer_format_name(desc.format)
                << " size_bytes must equal element_count * bytes_per_element; got size_bytes="
                << desc.size_bytes << ", element_count=" << desc.element_count
                << ", expected_size=" << expected_size;
            throw std::runtime_error(oss.str());
        }
        validate_format_support(desc, output, label);
    }

    std::string thread_vector_matrix_multiply_support_json(
        const char* name,
        D3D12_LINEAR_ALGEBRA_DATATYPE vector_input,
        D3D12_LINEAR_ALGEBRA_DATATYPE matrix_input,
        D3D12_LINEAR_ALGEBRA_DATATYPE bias_input,
        D3D12_LINEAR_ALGEBRA_DATATYPE vector_result) {
        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT support = {};
        support.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY;
        support.ThreadVectorMatrixMultiply.VectorInputType = vector_input;
        support.ThreadVectorMatrixMultiply.MatrixInputType = matrix_input;
        support.ThreadVectorMatrixMultiply.BiasInputType = bias_input;
        support.ThreadVectorMatrixMultiply.VectorResultType = vector_result;

        HRESULT hr = device->CheckFeatureSupport(
            D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
            &support,
            sizeof(support));

        std::ostringstream oss;
        oss << "{\"name\":\"" << name << "\",";
        oss << "\"operation\":\"thread_vector_matrix_multiply\",";
        oss << "\"query_ok\":" << (SUCCEEDED(hr) ? "true" : "false") << ",";
        oss << "\"query_hresult\":\"" << hresult_hex(hr) << "\",";
        oss << "\"vector_input_type\":\"" << linear_algebra_datatype_name(vector_input) << "\",";
        oss << "\"matrix_input_type\":\"" << linear_algebra_datatype_name(matrix_input) << "\",";
        oss << "\"bias_input_type\":\"" << linear_algebra_datatype_name(bias_input) << "\",";
        oss << "\"vector_result_type\":\"" << linear_algebra_datatype_name(vector_result) << "\",";
        oss << "\"support_flags\":" << linear_algebra_support_flags_json(support.ThreadVectorMatrixMultiply.SupportFlags);
        oss << "}";
        return oss.str();
    }

    std::string wave_matrix_multiply_support_json(
        const char* name,
        UINT wave_size,
        D3D12_LINEAR_ALGEBRA_DATATYPE matrix_a,
        D3D12_LINEAR_ALGEBRA_DATATYPE matrix_b,
        D3D12_LINEAR_ALGEBRA_DATATYPE accumulator) {
        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT support = {};
        support.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_WAVE_MATRIX_MULTIPLY;
        support.WaveMatrixMultiply.Inputs.WaveSize = wave_size;
        support.WaveMatrixMultiply.Inputs.MatrixAComponentType = matrix_a;
        support.WaveMatrixMultiply.Inputs.MatrixBComponentType = matrix_b;
        support.WaveMatrixMultiply.Inputs.AccumulatorComponentType = accumulator;

        HRESULT hr = device->CheckFeatureSupport(
            D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
            &support,
            sizeof(support));

        std::ostringstream oss;
        oss << "{\"name\":\"" << name << "\",";
        oss << "\"operation\":\"wave_matrix_multiply\",";
        oss << "\"query_ok\":" << (SUCCEEDED(hr) ? "true" : "false") << ",";
        oss << "\"query_hresult\":\"" << hresult_hex(hr) << "\",";
        oss << "\"wave_size\":" << wave_size << ",";
        oss << "\"matrix_a_type\":\"" << linear_algebra_datatype_name(matrix_a) << "\",";
        oss << "\"matrix_b_type\":\"" << linear_algebra_datatype_name(matrix_b) << "\",";
        oss << "\"accumulator_type\":\"" << linear_algebra_datatype_name(accumulator) << "\",";
        oss << "\"support_flags\":" << linear_algebra_support_flags_json(support.WaveMatrixMultiply.SupportFlags) << ",";
        oss << "\"num_shapes\":" << support.WaveMatrixMultiply.NumShapes;
        oss << "}";
        return oss.str();
    }

    std::string caps_json() {
        D3D12_FEATURE_DATA_SHADER_MODEL shader_model = { D3D_SHADER_MODEL_6_10 };
        HRESULT sm_hr = device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shader_model, sizeof(shader_model));
        const bool shader_model_query_ok = SUCCEEDED(sm_hr);

        D3D12_FEATURE_DATA_D3D12_OPTIONS9 options9 = {};
        HRESULT opt9_hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &options9, sizeof(options9));

        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT linear_algebra = {};
        HRESULT linear_algebra_hr = device->CheckFeatureSupport(
            D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT,
            &linear_algebra,
            sizeof(linear_algebra));

        UINT64 timestamp_frequency = 0;
        HRESULT freq_hr = queue->GetTimestampFrequency(&timestamp_frequency);

        std::ostringstream oss;
        oss << "{";
        oss << "\"status\":\"passed\",";
        oss << "\"adapter_name\":\"" << json_escape(adapter_info.name) << "\",";
        oss << "\"vendor_id\":" << adapter_info.vendor_id << ",";
        oss << "\"device_id\":" << adapter_info.device_id << ",";
        oss << "\"dedicated_video_memory\":" << adapter_info.dedicated_video_memory << ",";
        oss << "\"agility_sdk_version\":" << sdk_version_used << ",";
        oss << "\"agility_sdk_version_requested\":" << HLSL_PROBE_AGILITY_SDK_VERSION << ",";
        oss << "\"agility_sdk_path\":\"" << json_escape(sdk_path) << "\",";
        oss << "\"experimental_shader_models_global_ok\":" << (SUCCEEDED(global_experimental_shader_models_hr) ? "true" : "false") << ",";
        oss << "\"experimental_shader_models_global_hresult\":\"" << hresult_hex(global_experimental_shader_models_hr) << "\",";
        oss << "\"experimental_shader_models_factory_ok\":" << (SUCCEEDED(factory_experimental_shader_models_hr) ? "true" : "false") << ",";
        oss << "\"experimental_shader_models_factory_hresult\":\"" << hresult_hex(factory_experimental_shader_models_hr) << "\",";
        oss << "\"shader_model_query_ok\":" << (shader_model_query_ok ? "true" : "false") << ",";
        oss << "\"shader_model_query_hresult\":\"" << hresult_hex(sm_hr) << "\",";
        oss << "\"highest_shader_model\":\"" << shader_model_name(shader_model.HighestShaderModel) << "\",";
        oss << "\"supports_sm_6_10\":" << (shader_model_query_ok && shader_model.HighestShaderModel >= D3D_SHADER_MODEL_6_10 ? "true" : "false") << ",";
        oss << "\"supports_sm_6_9\":" << (shader_model_query_ok && shader_model.HighestShaderModel >= D3D_SHADER_MODEL_6_9 ? "true" : "false") << ",";
        oss << "\"options9_query_ok\":" << (SUCCEEDED(opt9_hr) ? "true" : "false") << ",";
        oss << "\"options9_query_hresult\":\"" << hresult_hex(opt9_hr) << "\",";
        oss << "\"wave_mma_tier\":" << static_cast<int>(options9.WaveMMATier) << ",";
        oss << "\"wave_mma_tier_name\":\"" << wave_mma_tier_name(options9.WaveMMATier) << "\",";
        oss << "\"linear_algebra_query_ok\":" << (SUCCEEDED(linear_algebra_hr) ? "true" : "false") << ",";
        oss << "\"linear_algebra_query_hresult\":\"" << hresult_hex(linear_algebra_hr) << "\",";
        oss << "\"linear_algebra_tier\":" << static_cast<int>(linear_algebra.LinearAlgebraTier) << ",";
        oss << "\"linear_algebra_tier_name\":\"" << linear_algebra_tier_name(linear_algebra.LinearAlgebraTier) << "\",";
        oss << "\"linear_algebra_thread_vector_matrix_multiply\":[";
        oss << thread_vector_matrix_multiply_support_json(
            "f16_vector_f16_matrix_f16_bias_f16_result",
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16);
        oss << "," << thread_vector_matrix_multiply_support_json(
            "f32_vector_f32_matrix_f32_bias_f32_result",
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32);
        oss << "," << thread_vector_matrix_multiply_support_json(
            "sint8_vector_sint8_matrix_sint32_bias_sint32_result",
            D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8,
            D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8,
            D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32,
            D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32);
        oss << "],";
        oss << "\"linear_algebra_wave_matrix_multiply\":[";
        oss << wave_matrix_multiply_support_json(
            "wave32_f16_f16_f32",
            32,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32);
        oss << "," << wave_matrix_multiply_support_json(
            "wave32_sint8_sint8_sint32",
            32,
            D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8,
            D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8,
            D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32);
        oss << "],";
        oss << "\"timestamp_frequency\":" << (SUCCEEDED(freq_hr) ? timestamp_frequency : 0);
        oss << "}";
        return oss.str();
    }

    std::string run_dxil(const void* dxil_data, uint64_t dxil_size, const HlslProbeRunConfig& config) {
        if (!dxil_data || dxil_size == 0) {
            throw std::runtime_error("DXIL bytecode is empty");
        }
        if (config.output_count > 0 && !config.outputs) {
            throw std::runtime_error("output_count is nonzero but outputs is null");
        }
        if (config.input_count > 0 && !config.inputs) {
            throw std::runtime_error("input_count is nonzero but inputs is null");
        }

        const uint32_t dispatch_x = config.dispatch_x ? config.dispatch_x : 1;
        const uint32_t dispatch_y = config.dispatch_y ? config.dispatch_y : 1;
        const uint32_t dispatch_z = config.dispatch_z ? config.dispatch_z : 1;

        std::vector<GpuBuffer> inputs(config.input_count);
        std::vector<GpuBuffer> outputs(config.output_count);

        throw_if_failed(allocator->Reset(), "ID3D12CommandAllocator::Reset");
        throw_if_failed(command_list->Reset(allocator.Get(), nullptr), "ID3D12GraphicsCommandList::Reset");

        for (uint32_t i = 0; i < config.input_count; ++i) {
            const auto& src = config.inputs[i];
            validate_buffer_desc(src.desc, false, i);
            if (!src.data && src.desc.size_bytes > 0) {
                throw std::runtime_error("input buffer data is null");
            }
            auto& dst = inputs[i];
            dst.desc = src.desc;
            dst.requested_size = src.desc.size_bytes;
            dst.resource_size = align_up(std::max<uint64_t>(src.desc.size_bytes, 4), 4);
            dst.resource = create_buffer(dst.resource_size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
            dst.upload = create_buffer(dst.resource_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped = nullptr;
            throw_if_failed(dst.upload->Map(0, nullptr, &mapped), "Map(input upload)");
            if (src.desc.size_bytes > 0) {
                std::memcpy(mapped, src.data, static_cast<size_t>(src.desc.size_bytes));
            }
            if (dst.resource_size > src.desc.size_bytes) {
                std::memset(static_cast<uint8_t*>(mapped) + src.desc.size_bytes, 0, static_cast<size_t>(dst.resource_size - src.desc.size_bytes));
            }
            dst.upload->Unmap(0, nullptr);
            command_list->CopyBufferRegion(dst.resource.Get(), 0, dst.upload.Get(), 0, dst.resource_size);
            transition(dst.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

        for (uint32_t i = 0; i < config.output_count; ++i) {
            const auto& out = config.outputs[i];
            validate_buffer_desc(out.desc, true, i);
            if (!out.data && out.desc.size_bytes > 0) {
                throw std::runtime_error("output buffer data is null");
            }
            auto& dst = outputs[i];
            dst.desc = out.desc;
            dst.requested_size = out.desc.size_bytes;
            dst.resource_size = align_up(std::max<uint64_t>(out.desc.size_bytes, 4), 4);
            dst.resource = create_buffer(dst.resource_size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            dst.readback = create_buffer(dst.resource_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        }

        D3D12_DESCRIPTOR_RANGE range_storage[2] = {};
        D3D12_ROOT_PARAMETER param_storage[2] = {};
        UINT root_param_count = 0;
        UINT srv_root_index = std::numeric_limits<UINT>::max();
        UINT uav_root_index = std::numeric_limits<UINT>::max();

        if (config.input_count > 0) {
            auto& range = range_storage[root_param_count];
            range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            range.NumDescriptors = config.input_count;
            range.BaseShaderRegister = 0;
            range.RegisterSpace = 0;
            range.OffsetInDescriptorsFromTableStart = 0;
            auto& param = param_storage[root_param_count];
            param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            param.DescriptorTable.NumDescriptorRanges = 1;
            param.DescriptorTable.pDescriptorRanges = &range;
            srv_root_index = root_param_count++;
        }

        if (config.output_count > 0) {
            auto& range = range_storage[root_param_count];
            range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            range.NumDescriptors = config.output_count;
            range.BaseShaderRegister = 0;
            range.RegisterSpace = 0;
            range.OffsetInDescriptorsFromTableStart = 0;
            auto& param = param_storage[root_param_count];
            param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            param.DescriptorTable.NumDescriptorRanges = 1;
            param.DescriptorTable.pDescriptorRanges = &range;
            uav_root_index = root_param_count++;
        }

        D3D12_ROOT_SIGNATURE_DESC root_desc = {};
        root_desc.NumParameters = root_param_count;
        root_desc.pParameters = root_param_count ? param_storage : nullptr;

        ComPtr<ID3DBlob> signature_blob;
        ComPtr<ID3DBlob> signature_error;
        HRESULT sig_hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature_blob, &signature_error);
        if (FAILED(sig_hr)) {
            std::string error_text;
            if (signature_error) {
                error_text.assign(static_cast<const char*>(signature_error->GetBufferPointer()), signature_error->GetBufferSize());
            }
            throw std::runtime_error("D3D12SerializeRootSignature failed: " + error_text + " (" + hresult_hex(sig_hr) + ")");
        }

        ComPtr<ID3D12RootSignature> root_signature;
        throw_if_failed(device->CreateRootSignature(0, signature_blob->GetBufferPointer(), signature_blob->GetBufferSize(), IID_PPV_ARGS(&root_signature)), "CreateRootSignature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
        pso_desc.pRootSignature = root_signature.Get();
        pso_desc.CS.pShaderBytecode = dxil_data;
        pso_desc.CS.BytecodeLength = static_cast<SIZE_T>(dxil_size);

        ComPtr<ID3D12PipelineState> pso;
        throw_if_failed(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso)), "CreateComputePipelineState");

        ComPtr<ID3D12DescriptorHeap> descriptor_heap;
        const UINT descriptor_count = config.input_count + config.output_count;
        if (descriptor_count > 0) {
            D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
            heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heap_desc.NumDescriptors = descriptor_count;
            heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            throw_if_failed(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptor_heap)), "CreateDescriptorHeap");

            const UINT descriptor_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_CPU_DESCRIPTOR_HANDLE cpu = descriptor_heap->GetCPUDescriptorHandleForHeapStart();

            for (uint32_t i = 0; i < config.input_count; ++i) {
                const auto& desc = inputs[i].desc;
                D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
                srv_desc.Format = dxgi_format_for(desc.format);
                srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv_desc.Buffer.NumElements = static_cast<UINT>(desc.element_count);
                srv_desc.Buffer.Flags = desc.view_kind == HLSL_PROBE_BUFFER_VIEW_RAW ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
                device->CreateShaderResourceView(inputs[i].resource.Get(), &srv_desc, cpu);
                cpu.ptr += descriptor_size;
            }

            for (uint32_t i = 0; i < config.output_count; ++i) {
                const auto& desc = outputs[i].desc;
                D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
                uav_desc.Format = dxgi_format_for(desc.format);
                uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uav_desc.Buffer.NumElements = static_cast<UINT>(desc.element_count);
                uav_desc.Buffer.Flags = desc.view_kind == HLSL_PROBE_BUFFER_VIEW_RAW ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE;
                device->CreateUnorderedAccessView(outputs[i].resource.Get(), nullptr, &uav_desc, cpu);
                cpu.ptr += descriptor_size;
            }
        }

        D3D12_QUERY_HEAP_DESC query_desc = {};
        query_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        query_desc.Count = 2;
        ComPtr<ID3D12QueryHeap> query_heap;
        throw_if_failed(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&query_heap)), "CreateQueryHeap(timestamp)");
        auto timestamp_readback = create_buffer(sizeof(uint64_t) * 2, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

        command_list->SetPipelineState(pso.Get());
        command_list->SetComputeRootSignature(root_signature.Get());
        if (descriptor_heap) {
            ID3D12DescriptorHeap* heaps[] = { descriptor_heap.Get() };
            command_list->SetDescriptorHeaps(1, heaps);
            const UINT descriptor_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_GPU_DESCRIPTOR_HANDLE gpu = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
            if (srv_root_index != std::numeric_limits<UINT>::max()) {
                command_list->SetComputeRootDescriptorTable(srv_root_index, gpu);
            }
            if (uav_root_index != std::numeric_limits<UINT>::max()) {
                D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = gpu;
                uav_gpu.ptr += static_cast<UINT64>(config.input_count) * descriptor_size;
                command_list->SetComputeRootDescriptorTable(uav_root_index, uav_gpu);
            }
        }

        command_list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        command_list->Dispatch(dispatch_x, dispatch_y, dispatch_z);
        command_list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);

        if (!outputs.empty()) {
            D3D12_RESOURCE_BARRIER uav_barrier = {};
            uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            command_list->ResourceBarrier(1, &uav_barrier);
        }

        for (auto& out : outputs) {
            transition(out.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            command_list->CopyBufferRegion(out.readback.Get(), 0, out.resource.Get(), 0, out.resource_size);
        }

        command_list->ResolveQueryData(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, timestamp_readback.Get(), 0);
        throw_if_failed(command_list->Close(), "ID3D12GraphicsCommandList::Close");
        ID3D12CommandList* lists[] = { command_list.Get() };
        queue->ExecuteCommandLists(1, lists);
        wait_for_gpu();

        for (uint32_t i = 0; i < config.output_count; ++i) {
            void* mapped = nullptr;
            D3D12_RANGE read_range = { 0, static_cast<SIZE_T>(outputs[i].requested_size) };
            throw_if_failed(outputs[i].readback->Map(0, &read_range, &mapped), "Map(output readback)");
            if (outputs[i].requested_size > 0) {
                std::memcpy(config.outputs[i].data, mapped, static_cast<size_t>(outputs[i].requested_size));
            }
            D3D12_RANGE write_range = { 0, 0 };
            outputs[i].readback->Unmap(0, &write_range);
        }

        uint64_t timestamps[2] = {};
        void* timestamp_mapped = nullptr;
        D3D12_RANGE timestamp_range = { 0, sizeof(timestamps) };
        throw_if_failed(timestamp_readback->Map(0, &timestamp_range, &timestamp_mapped), "Map(timestamp readback)");
        std::memcpy(timestamps, timestamp_mapped, sizeof(timestamps));
        D3D12_RANGE timestamp_write_range = { 0, 0 };
        timestamp_readback->Unmap(0, &timestamp_write_range);

        UINT64 frequency = 0;
        queue->GetTimestampFrequency(&frequency);
        double gpu_time_ms = 0.0;
        if (frequency > 0 && timestamps[1] >= timestamps[0]) {
            gpu_time_ms = static_cast<double>(timestamps[1] - timestamps[0]) * 1000.0 / static_cast<double>(frequency);
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "{";
        oss << "\"status\":\"passed\",";
        oss << "\"dispatch\":[" << dispatch_x << "," << dispatch_y << "," << dispatch_z << "],";
        oss << "\"input_count\":" << config.input_count << ",";
        oss << "\"output_count\":" << config.output_count << ",";
        oss << "\"gpu_time_ms\":" << gpu_time_ms << ",";
        oss << "\"timestamp_frequency\":" << frequency << ",";
        oss << "\"timestamp_start\":" << timestamps[0] << ",";
        oss << "\"timestamp_end\":" << timestamps[1] << ",";
        oss << "\"output_sizes\":[";
        for (uint32_t i = 0; i < config.output_count; ++i) {
            if (i) oss << ",";
            oss << outputs[i].requested_size;
        }
        oss << "],";
        oss << "\"inputs\":[";
        for (uint32_t i = 0; i < config.input_count; ++i) {
            if (i) oss << ",";
            oss << buffer_desc_json(inputs[i].desc);
        }
        oss << "],\"outputs\":[";
        for (uint32_t i = 0; i < config.output_count; ++i) {
            if (i) oss << ",";
            oss << buffer_desc_json(outputs[i].desc);
        }
        oss << "]}";
        return oss.str();
    }
};

extern "C" HLSL_PROBE_API HlslProbeHandle hlsl_probe_create(const char* agility_sdk_path_utf8) {
    try {
        g_last_error.clear();
        std::string sdk_path = agility_sdk_path_utf8 ? agility_sdk_path_utf8 : "";
        return new HlslProbeContext(std::move(sdk_path));
    } catch (const std::exception& ex) {
        g_last_error = ex.what();
        return nullptr;
    } catch (...) {
        g_last_error = "unknown exception in hlsl_probe_create";
        return nullptr;
    }
}

extern "C" HLSL_PROBE_API void hlsl_probe_destroy(HlslProbeHandle handle) {
    delete handle;
}

extern "C" HLSL_PROBE_API const char* hlsl_probe_get_last_error(HlslProbeHandle handle) {
    return handle ? handle->last_error.c_str() : g_last_error.c_str();
}

extern "C" HLSL_PROBE_API const char* hlsl_probe_get_global_last_error(void) {
    return g_last_error.c_str();
}

extern "C" HLSL_PROBE_API char* hlsl_probe_get_caps_json(HlslProbeHandle handle) {
    if (!handle) {
        g_last_error = "hlsl_probe_get_caps_json called with null handle";
        return nullptr;
    }
    try {
        handle->last_error.clear();
        return copy_to_c_string(handle->caps_json());
    } catch (const std::exception& ex) {
        handle->last_error = ex.what();
        return nullptr;
    } catch (...) {
        handle->last_error = "unknown exception in hlsl_probe_get_caps_json";
        return nullptr;
    }
}

extern "C" HLSL_PROBE_API int hlsl_probe_run_dxil(
    HlslProbeHandle handle,
    const void* dxil_data,
    uint64_t dxil_size_bytes,
    const HlslProbeRunConfig* config,
    char** result_json) {
    if (result_json) {
        *result_json = nullptr;
    }
    if (!handle) {
        g_last_error = "hlsl_probe_run_dxil called with null handle";
        return 0;
    }
    if (!config) {
        handle->last_error = "hlsl_probe_run_dxil called with null config";
        return 0;
    }
    try {
        handle->last_error.clear();
        std::string result = handle->run_dxil(dxil_data, dxil_size_bytes, *config);
        if (result_json) {
            *result_json = copy_to_c_string(result);
            if (!*result_json) {
                throw std::runtime_error("failed to allocate result JSON");
            }
        }
        return 1;
    } catch (const std::exception& ex) {
        handle->last_error = ex.what();
        return 0;
    } catch (...) {
        handle->last_error = "unknown exception in hlsl_probe_run_dxil";
        return 0;
    }
}

extern "C" HLSL_PROBE_API void hlsl_probe_free_string(char* value) {
    std::free(value);
}
