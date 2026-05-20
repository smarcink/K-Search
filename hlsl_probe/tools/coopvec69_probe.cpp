#include <windows.h>
#include <initguid.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = 717;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}

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

const char* shader_model_name(D3D_SHADER_MODEL shader_model) {
    switch (shader_model) {
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

const char* cooperative_vector_tier_name(D3D12_COOPERATIVE_VECTOR_TIER tier) {
    switch (tier) {
    case D3D12_COOPERATIVE_VECTOR_TIER_NOT_SUPPORTED: return "not_supported";
    case D3D12_COOPERATIVE_VECTOR_TIER_1_0: return "1_0";
    case D3D12_COOPERATIVE_VECTOR_TIER_1_1: return "1_1";
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
    case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8_T4_PACKED: return "sint8_t4_packed";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8_T4_PACKED: return "uint8_t4_packed";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8: return "uint8";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8: return "sint8";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT_E4M3: return "float_e4m3";
    case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT_E5M2: return "float_e5m2";
    default: return "unknown";
    }
}

void write_datatype_json_field(const char* name, D3D12_LINEAR_ALGEBRA_DATATYPE datatype) {
    std::cout << "\"" << name << "\":\"" << linear_algebra_datatype_name(datatype) << "\",";
    std::cout << "\"" << name << "_value\":" << static_cast<int>(datatype);
}

std::vector<uint8_t> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) {
        return {};
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    return bytes;
}

std::string executable_d3d12_path() {
    char path[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameA(nullptr, path, static_cast<DWORD>(sizeof(path)));
    if (length == 0 || length >= sizeof(path)) {
        return ".\\D3D12\\";
    }
    std::string module_path(path, length);
    const size_t slash = module_path.find_last_of("\\/");
    if (slash == std::string::npos) {
        return ".\\D3D12\\";
    }
    return module_path.substr(0, slash + 1) + "D3D12\\";
}

HRESULT create_compute_pso(ID3D12Device* device, const std::vector<uint8_t>& dxil) {
    D3D12_DESCRIPTOR_RANGE ranges[3] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 1;
    ranges[1].RegisterSpace = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 0;
    ranges[2].RegisterSpace = 0;
    ranges[2].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER root_params[3] = {};
    for (UINT i = 0; i < 3; ++i) {
        root_params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_params[i].DescriptorTable.NumDescriptorRanges = 1;
        root_params[i].DescriptorTable.pDescriptorRanges = &ranges[i];
        root_params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_ROOT_SIGNATURE_DESC root_desc = {};
    root_desc.NumParameters = 3;
    root_desc.pParameters = root_params;
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> error_blob;
    HRESULT hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &error_blob);
    if (FAILED(hr)) {
        return hr;
    }

    ComPtr<ID3D12RootSignature> root_signature;
    hr = device->CreateRootSignature(
        0,
        root_blob->GetBufferPointer(),
        root_blob->GetBufferSize(),
        IID_PPV_ARGS(&root_signature));
    if (FAILED(hr)) {
        return hr;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.pRootSignature = root_signature.Get();
    pso_desc.CS.pShaderBytecode = dxil.data();
    pso_desc.CS.BytecodeLength = dxil.size();

    ComPtr<ID3D12PipelineState> pso;
    return device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso));
}

int main(int argc, char** argv) {
    const char* dxil_path = nullptr;
    bool request_coop_experiment = true;
    for (int arg = 1; arg < argc; ++arg) {
        if (std::string(argv[arg]) == "--shader-models-only") {
            request_coop_experiment = false;
        } else {
            dxil_path = argv[arg];
        }
    }

    const UUID features[] = { D3D12ExperimentalShaderModels, D3D12CooperativeVectorExperiment };
    const HRESULT experimental_hr = D3D12EnableExperimentalFeatures(
        request_coop_experiment ? 2 : 1,
        features,
        nullptr,
        nullptr);

    ComPtr<IDXGIFactory6> factory;
    HRESULT factory_hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));

    ComPtr<IDXGIAdapter1> adapter;
    std::string adapter_name;
    if (SUCCEEDED(factory_hr)) {
        for (UINT index = 0; ; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            HRESULT enum_hr = factory->EnumAdapterByGpuPreference(
                index,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&candidate));
            if (enum_hr == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (FAILED(enum_hr)) {
                continue;
            }
            DXGI_ADAPTER_DESC1 desc = {};
            candidate->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                continue;
            }
            adapter = candidate;
            adapter_name = wide_to_utf8(desc.Description);
            break;
        }
    }

    ComPtr<ID3D12Device> device;
    HRESULT global_device_hr = adapter
        ? D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))
        : E_FAIL;

    ComPtr<ID3D12Device> factory_device;
    ComPtr<ID3D12SDKConfiguration1> sdk_config;
    HRESULT sdk_config_hr = D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&sdk_config));

    ComPtr<ID3D12DeviceFactory> device_factory;
    const std::string d3d12_path = executable_d3d12_path();
    HRESULT create_device_factory_hr = sdk_config
        ? sdk_config->CreateDeviceFactory(717, d3d12_path.c_str(), IID_PPV_ARGS(&device_factory))
        : E_FAIL;
    HRESULT factory_experimental_hr = device_factory
        ? device_factory->EnableExperimentalFeatures(request_coop_experiment ? 2 : 1, features, nullptr, nullptr)
        : E_FAIL;
    HRESULT factory_device_hr = adapter && device_factory
        ? device_factory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&factory_device))
        : E_FAIL;

    const char* device_create_path = "global";
    HRESULT device_hr = global_device_hr;
    if (factory_device) {
        device = factory_device;
        device_hr = factory_device_hr;
        device_create_path = "device_factory";
    }

    D3D12_FEATURE_DATA_SHADER_MODEL shader_model = { D3D_SHADER_MODEL_6_9 };
    HRESULT shader_model_hr = device
        ? device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shader_model, sizeof(shader_model))
        : E_FAIL;

    D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL experimental_options = {};
    HRESULT coop_tier_hr = device
        ? device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL,
            &experimental_options,
            sizeof(experimental_options))
        : E_FAIL;

    D3D12_FEATURE_DATA_COOPERATIVE_VECTOR coop_props = {};
    HRESULT coop_props_count_hr = device
        ? device->CheckFeatureSupport(D3D12_FEATURE_COOPERATIVE_VECTOR, &coop_props, sizeof(coop_props))
        : E_FAIL;

    std::vector<D3D12_COOPERATIVE_VECTOR_PROPERTIES_MUL> mul_props(coop_props.MatrixVectorMulAddPropCount);
    std::vector<D3D12_COOPERATIVE_VECTOR_PROPERTIES_ACCUMULATE> outer_props(coop_props.OuterProductAccumulatePropCount);
    std::vector<D3D12_COOPERATIVE_VECTOR_PROPERTIES_ACCUMULATE> vector_props(coop_props.VectorAccumulatePropCount);
    if (SUCCEEDED(coop_props_count_hr)) {
        coop_props.pMatrixVectorMulAddProperties = mul_props.empty() ? nullptr : mul_props.data();
        coop_props.pOuterProductAccumulateProperties = outer_props.empty() ? nullptr : outer_props.data();
        coop_props.pVectorAccumulateProperties = vector_props.empty() ? nullptr : vector_props.data();
    }
    HRESULT coop_props_fill_hr = device && SUCCEEDED(coop_props_count_hr)
        ? device->CheckFeatureSupport(D3D12_FEATURE_COOPERATIVE_VECTOR, &coop_props, sizeof(coop_props))
        : E_FAIL;

    HRESULT pso_hr = E_FAIL;
    size_t dxil_size = 0;
    if (device && dxil_path) {
        std::vector<uint8_t> dxil = read_file(dxil_path);
        dxil_size = dxil.size();
        if (!dxil.empty()) {
            pso_hr = create_compute_pso(device.Get(), dxil);
        }
    }

    std::cout << "{\n";
    std::cout << "  \"agility_sdk_version_requested\": 717,\n";
    std::cout << "  \"coop_experiment_requested\": " << (request_coop_experiment ? "true" : "false") << ",\n";
    std::cout << "  \"experimental_features_hresult\": \"" << hresult_hex(experimental_hr) << "\",\n";
    std::cout << "  \"experimental_features_ok\": " << (SUCCEEDED(experimental_hr) ? "true" : "false") << ",\n";
    std::cout << "  \"factory_hresult\": \"" << hresult_hex(factory_hr) << "\",\n";
    std::cout << "  \"sdk_config_hresult\": \"" << hresult_hex(sdk_config_hr) << "\",\n";
    std::cout << "  \"create_device_factory_hresult\": \"" << hresult_hex(create_device_factory_hr) << "\",\n";
    std::cout << "  \"factory_experimental_features_hresult\": \"" << hresult_hex(factory_experimental_hr) << "\",\n";
    std::cout << "  \"global_device_hresult\": \"" << hresult_hex(global_device_hr) << "\",\n";
    std::cout << "  \"factory_device_hresult\": \"" << hresult_hex(factory_device_hr) << "\",\n";
    std::cout << "  \"device_create_path\": \"" << device_create_path << "\",\n";
    std::cout << "  \"adapter_name\": \"" << adapter_name << "\",\n";
    std::cout << "  \"device_hresult\": \"" << hresult_hex(device_hr) << "\",\n";
    std::cout << "  \"shader_model_query_hresult\": \"" << hresult_hex(shader_model_hr) << "\",\n";
    std::cout << "  \"highest_shader_model\": \"" << shader_model_name(shader_model.HighestShaderModel) << "\",\n";
    std::cout << "  \"supports_sm_6_9\": " << (SUCCEEDED(shader_model_hr) && shader_model.HighestShaderModel >= D3D_SHADER_MODEL_6_9 ? "true" : "false") << ",\n";
    std::cout << "  \"cooperative_vector_tier_hresult\": \"" << hresult_hex(coop_tier_hr) << "\",\n";
    std::cout << "  \"cooperative_vector_tier\": " << static_cast<int>(experimental_options.CooperativeVectorTier) << ",\n";
    std::cout << "  \"cooperative_vector_tier_name\": \"" << cooperative_vector_tier_name(experimental_options.CooperativeVectorTier) << "\",\n";
    std::cout << "  \"cooperative_vector_properties_count_hresult\": \"" << hresult_hex(coop_props_count_hr) << "\",\n";
    std::cout << "  \"cooperative_vector_properties_fill_hresult\": \"" << hresult_hex(coop_props_fill_hr) << "\",\n";
    std::cout << "  \"matrix_vector_mul_add_property_count\": " << coop_props.MatrixVectorMulAddPropCount << ",\n";
    std::cout << "  \"outer_product_accumulate_property_count\": " << coop_props.OuterProductAccumulatePropCount << ",\n";
    std::cout << "  \"vector_accumulate_property_count\": " << coop_props.VectorAccumulatePropCount << ",\n";
    std::cout << "  \"matrix_vector_mul_add_properties\": [";
    for (size_t i = 0; i < mul_props.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        const D3D12_COOPERATIVE_VECTOR_PROPERTIES_MUL& prop = mul_props[i];
        std::cout << "{\"index\":" << i << ",";
        write_datatype_json_field("input_type", prop.InputType);
        std::cout << ",";
        write_datatype_json_field("input_interpretation", prop.InputInterpretation);
        std::cout << ",";
        write_datatype_json_field("matrix_interpretation", prop.MatrixInterpretation);
        std::cout << ",";
        write_datatype_json_field("bias_interpretation", prop.BiasInterpretation);
        std::cout << ",";
        write_datatype_json_field("output_type", prop.OutputType);
        std::cout << ",\"transpose_supported\":" << (prop.TransposeSupported ? "true" : "false") << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"outer_product_accumulate_properties\": [";
    for (size_t i = 0; i < outer_props.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        const D3D12_COOPERATIVE_VECTOR_PROPERTIES_ACCUMULATE& prop = outer_props[i];
        std::cout << "{\"index\":" << i << ",";
        write_datatype_json_field("input_type", prop.InputType);
        std::cout << ",";
        write_datatype_json_field("accumulation_type", prop.AccumulationType);
        std::cout << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"vector_accumulate_properties\": [";
    for (size_t i = 0; i < vector_props.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        const D3D12_COOPERATIVE_VECTOR_PROPERTIES_ACCUMULATE& prop = vector_props[i];
        std::cout << "{\"index\":" << i << ",";
        write_datatype_json_field("input_type", prop.InputType);
        std::cout << ",";
        write_datatype_json_field("accumulation_type", prop.AccumulationType);
        std::cout << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"dxil_path\": \"" << (dxil_path ? dxil_path : "") << "\",\n";
    std::cout << "  \"dxil_size\": " << dxil_size << ",\n";
    std::cout << "  \"compute_pso_hresult\": \"" << hresult_hex(pso_hr) << "\",\n";
    std::cout << "  \"compute_pso_ok\": " << (SUCCEEDED(pso_hr) ? "true" : "false") << "\n";
    std::cout << "}\n";

    return (SUCCEEDED(experimental_hr) && SUCCEEDED(device_hr) && SUCCEEDED(coop_tier_hr)) ? 0 : 1;
}
