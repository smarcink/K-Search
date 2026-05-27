#include "hlsl_probe_c_api.h"
#include "hlsl_probe_config.h"

#include <windows.h>
#include <d3d12.h>

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = HLSL_PROBE_AGILITY_SDK_VERSION;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}

namespace {

enum class JsonOutputMode {
    Pretty,
    Compact,
};

std::string quote_arg(const std::string& value) {
    std::string out = "\"";
    for (char ch : value) {
        out += (ch == '"') ? "\\\"" : std::string(1, ch);
    }
    out += "\"";
    return out;
}

std::wstring utf8_to_wide(const std::string& value) {
    if (value.empty()) {
        return {};
    }
    int required = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0);
    if (required <= 0) {
        throw std::runtime_error("MultiByteToWideChar failed");
    }
    std::wstring output(static_cast<size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), output.data(), required);
    return output;
}

std::string windows_error_message(DWORD error) {
    LPSTR message = nullptr;
    DWORD size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&message),
        0,
        nullptr);
    std::string output = size && message ? std::string(message, size) : "unknown Windows error";
    if (message) {
        LocalFree(message);
    }
    return output;
}

std::string json_escape(const std::string& value) {
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
                const char* digits = "0123456789abcdef";
                output += "\\u00";
                output += digits[(static_cast<unsigned char>(ch) >> 4) & 0xf];
                output += digits[static_cast<unsigned char>(ch) & 0xf];
            } else {
                output += ch;
            }
            break;
        }
    }
    return output;
}

std::string pretty_json(const std::string& value) {
    std::string output;
    output.reserve(value.size() + value.size() / 4);

    int indent = 0;
    bool in_string = false;
    bool escaped = false;

    auto append_indent = [&]() {
        output.append(static_cast<size_t>(indent) * 2, ' ');
    };

    for (char ch : value) {
        if (in_string) {
            output += ch;
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        switch (ch) {
        case '"':
            in_string = true;
            output += ch;
            break;
        case '{':
        case '[':
            output += ch;
            output += '\n';
            ++indent;
            append_indent();
            break;
        case '}':
        case ']':
            output += '\n';
            if (indent > 0) {
                --indent;
            }
            append_indent();
            output += ch;
            break;
        case ',':
            output += ch;
            output += '\n';
            append_indent();
            break;
        case ':':
            output += ": ";
            break;
        default:
            if (!std::isspace(static_cast<unsigned char>(ch))) {
                output += ch;
            }
            break;
        }
    }

    return output;
}

std::string classify_run_error(const std::string& error) {
    if (error.find("CreateComputePipelineState") != std::string::npos) {
        return "pso";
    }
    if (error.find("Dispatch") != std::string::npos || error.find("ExecuteCommandLists") != std::string::npos || error.find("Signal") != std::string::npos) {
        return "dispatch";
    }
    if (error.find("readback") != std::string::npos || error.find("Map(output") != std::string::npos) {
        return "readback";
    }
    return "run";
}

std::string error_json(const std::string& stage, const std::string& target, const std::string& error) {
    std::ostringstream oss;
    oss << "{\"status\":\"failed\",\"stage\":\"" << json_escape(stage)
        << "\",\"target\":\"" << json_escape(target)
        << "\",\"error\":\"" << json_escape(error) << "\"}";
    return oss.str();
}

std::filesystem::path temp_file_path(const wchar_t* prefix, const wchar_t* extension) {
    std::wstring temp_dir(MAX_PATH, L'\0');
    DWORD dir_len = GetTempPathW(static_cast<DWORD>(temp_dir.size()), temp_dir.data());
    temp_dir.resize(dir_len);

    std::wstring temp_name(MAX_PATH, L'\0');
    if (!GetTempFileNameW(temp_dir.c_str(), prefix, 0, temp_name.data())) {
        throw std::runtime_error("GetTempFileNameW failed");
    }
    std::filesystem::path path(temp_name.c_str());
    std::filesystem::path renamed = path;
    renamed.replace_extension(extension);
    std::error_code ec;
    std::filesystem::rename(path, renamed, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        return path.replace_extension(extension);
    }
    return renamed;
}

std::filesystem::path executable_directory() {
    std::wstring buffer(MAX_PATH, L'\0');
    while (true) {
        DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
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

std::filesystem::path find_existing_repo_path(const std::filesystem::path& relative_path) {
    std::vector<std::filesystem::path> roots;
    roots.push_back(std::filesystem::current_path());

    std::filesystem::path exe_dir = executable_directory();
    for (std::filesystem::path cursor = exe_dir; !cursor.empty(); cursor = cursor.parent_path()) {
        roots.push_back(cursor);
        if (cursor == cursor.root_path()) {
            break;
        }
    }

    for (const auto& root : roots) {
        std::filesystem::path candidate = root / relative_path;
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

std::filesystem::path resolve_dxc_path() {
    std::filesystem::path configured = HLSL_PROBE_DEFAULT_DXC_PATH;
    if (std::filesystem::exists(configured)) {
        return configured;
    }
    return find_existing_repo_path("thirdparty/dxc_preview_2026_04_22/bin/x64/dxc.exe");
}

std::filesystem::path resolve_hlsl_include_path() {
    std::filesystem::path configured = HLSL_PROBE_DEFAULT_HLSL_INCLUDE;
    if (std::filesystem::exists(configured)) {
        return configured;
    }
    return find_existing_repo_path("thirdparty/dxc_preview_2026_04_22/inc/hlsl");
}

int run_command_capture(const std::string& command, std::string& output) {
    SECURITY_ATTRIBUTES security = {};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;

    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &security, 0)) {
        output = "CreatePipe failed: " + windows_error_message(GetLastError());
        return -1;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW startup = {};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_pipe;
    startup.hStdError = write_pipe;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION process = {};
    std::wstring command_line = utf8_to_wide(command);
    BOOL created = CreateProcessW(
        nullptr,
        command_line.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &startup,
        &process);
    CloseHandle(write_pipe);

    if (!created) {
        output = "CreateProcessW failed: " + windows_error_message(GetLastError()) + "\nCommand: " + command;
        CloseHandle(read_pipe);
        return -1;
    }

    char buffer[4096];
    DWORD bytes_read = 0;
    while (ReadFile(read_pipe, buffer, sizeof(buffer), &bytes_read, nullptr) && bytes_read > 0) {
        output.append(buffer, buffer + bytes_read);
    }

    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(read_pipe);
    return static_cast<int>(exit_code);
}

std::vector<uint8_t> read_binary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file for reading: " + path.string());
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open file for writing: " + path.string());
    }
    out << text;
}

std::vector<uint8_t> compile_hlsl(const std::string& hlsl, const std::string& target = "cs_6_8") {
    const auto hlsl_path = temp_file_path(L"hpr", L".hlsl");
    const auto dxil_path = temp_file_path(L"hpr", L".dxil");
    write_text(hlsl_path, hlsl);

    const auto dxc_path = resolve_dxc_path();
    const auto include_path = resolve_hlsl_include_path();
    if (dxc_path.empty() || include_path.empty()) {
        std::filesystem::remove(hlsl_path);
        std::filesystem::remove(dxil_path);
        std::ostringstream oss;
        oss << "DXC toolchain not found. Configured dxc path: " << HLSL_PROBE_DEFAULT_DXC_PATH
            << "; configured HLSL include path: " << HLSL_PROBE_DEFAULT_HLSL_INCLUDE
            << "; also searched from current directory and executable parents.";
        throw std::runtime_error(oss.str());
    }

    std::ostringstream cmd;
    cmd << quote_arg(dxc_path.string())
        << " -T " << target
        << " -E main"
        << " -HV 2021"
        << " -enable-16bit-types"
        << " -I " << quote_arg(include_path.string())
        << " -Fo " << quote_arg(dxil_path.string())
        << " " << quote_arg(hlsl_path.string());

    std::string output;
    int rc = run_command_capture(cmd.str(), output);
    if (rc != 0) {
        std::filesystem::remove(hlsl_path);
        std::filesystem::remove(dxil_path);
        throw std::runtime_error("dxc failed:\nCommand: " + cmd.str() + "\nOutput:\n" + output);
    }

    auto dxil = read_binary(dxil_path);
    std::filesystem::remove(hlsl_path);
    std::filesystem::remove(dxil_path);
    return dxil;
}

HlslProbeHandle create_probe_or_throw() {
    HlslProbeHandle handle = hlsl_probe_create(nullptr);
    if (!handle) {
        throw std::runtime_error(hlsl_probe_get_global_last_error());
    }
    return handle;
}

HlslProbeBufferDesc raw_u32_desc(uint64_t size_bytes) {
    HlslProbeBufferDesc desc = {};
    desc.struct_size = sizeof(HlslProbeBufferDesc);
    desc.view_kind = HLSL_PROBE_BUFFER_VIEW_RAW;
    desc.format = HLSL_PROBE_BUFFER_FORMAT_RAW_U32;
    desc.element_count = size_bytes / 4;
    desc.size_bytes = size_bytes;
    return desc;
}

void print_json_and_free(char* text, JsonOutputMode output_mode) {
    if (text) {
        std::cout << (output_mode == JsonOutputMode::Compact ? text : pretty_json(text)) << std::endl;
        hlsl_probe_free_string(text);
    }
}

int command_probe(JsonOutputMode output_mode) {
    HlslProbeHandle handle = create_probe_or_throw();
    char* caps = hlsl_probe_get_caps_json(handle);
    if (!caps) {
        std::cerr << hlsl_probe_get_last_error(handle) << std::endl;
        hlsl_probe_destroy(handle);
        return 1;
    }
    print_json_and_free(caps, output_mode);
    hlsl_probe_destroy(handle);
    return 0;
}

int command_self_test() {
    static const char* shader = R"HLSL(
RWByteAddressBuffer Output : register(u0);

[numthreads(1, 1, 1)]
void main() {
    Output.Store(0, 123u);
}
)HLSL";

    auto dxil = compile_hlsl(shader);
    HlslProbeHandle handle = create_probe_or_throw();
    uint32_t value = 0;
    HlslProbeOutputBuffer output = { &value, raw_u32_desc(sizeof(value)) };
    HlslProbeRunConfig config = {};
    config.dispatch_x = 1;
    config.dispatch_y = 1;
    config.dispatch_z = 1;
    config.outputs = &output;
    config.output_count = 1;

    char* run_json = nullptr;
    int ok = hlsl_probe_run_dxil(handle, dxil.data(), dxil.size(), &config, &run_json);
    if (!ok) {
        std::cerr << hlsl_probe_get_last_error(handle) << std::endl;
        hlsl_probe_destroy(handle);
        return 1;
    }

    std::cout << "{\"status\":\"" << (value == 123 ? "passed" : "failed")
              << "\",\"output_value\":" << value
              << ",\"run\":" << (run_json ? run_json : "null") << "}" << std::endl;
    hlsl_probe_free_string(run_json);
    hlsl_probe_destroy(handle);
    return value == 123 ? 0 : 2;
}

int command_linalg_test(const std::string& target) {
    static const char* shader = R"HLSL(
#include <dx/linalg.h>

ByteAddressBuffer MatrixData : register(t0);
RWByteAddressBuffer Output : register(u0);

[numthreads(1, 1, 1)]
void main() {
    dx::linalg::Matrix<dx::linalg::ComponentType::U32, 2, 2, dx::linalg::MatrixUse::A, dx::linalg::MatrixScope::Thread> matrix = dx::linalg::Matrix<dx::linalg::ComponentType::U32, 2, 2, dx::linalg::MatrixUse::A, dx::linalg::MatrixScope::Thread>::Load<dx::linalg::MatrixLayout::RowMajor>(MatrixData, 0, 8);
    vector<uint32_t, 2> input = { 5u, 6u };
    vector<uint32_t, 2> result = dx::linalg::Multiply<uint32_t>(matrix, input);
    Output.Store(0, result.x);
    Output.Store(4, result.y);
}
)HLSL";

    std::vector<uint8_t> dxil;
    try {
        dxil = compile_hlsl(shader, target);
    } catch (const std::exception& ex) {
        std::cout << error_json("compile", target, ex.what()) << std::endl;
        return 1;
    }

    HlslProbeHandle handle = nullptr;
    try {
        handle = create_probe_or_throw();
    } catch (const std::exception& ex) {
        std::cout << error_json("device", target, ex.what()) << std::endl;
        return 1;
    }

    uint32_t matrix_data[4] = { 1u, 2u, 3u, 4u };
    uint32_t output_values[2] = { 0u, 0u };
    HlslProbeInputBuffer input = { matrix_data, raw_u32_desc(sizeof(matrix_data)) };
    HlslProbeOutputBuffer output = { output_values, raw_u32_desc(sizeof(output_values)) };
    HlslProbeRunConfig config = {};
    config.dispatch_x = 1;
    config.dispatch_y = 1;
    config.dispatch_z = 1;
    config.inputs = &input;
    config.input_count = 1;
    config.outputs = &output;
    config.output_count = 1;

    char* run_json = nullptr;
    int ok = hlsl_probe_run_dxil(handle, dxil.data(), dxil.size(), &config, &run_json);
    if (!ok) {
        std::string error = hlsl_probe_get_last_error(handle);
        hlsl_probe_destroy(handle);
        std::cout << error_json(classify_run_error(error), target, error) << std::endl;
        return 1;
    }

    constexpr uint32_t expected0 = 17u;
    constexpr uint32_t expected1 = 39u;
    const bool passed = output_values[0] == expected0 && output_values[1] == expected1;
    std::cout << "{\"status\":\"" << (passed ? "passed" : "failed")
              << "\",\"stage\":\"" << (passed ? "dispatch" : "verify")
              << "\",\"target\":\"" << json_escape(target)
              << "\",\"dxil_size\":" << dxil.size()
              << ",\"expected_values\":[" << expected0 << "," << expected1 << "]"
              << ",\"output_values\":[" << output_values[0] << "," << output_values[1] << "]"
              << ",\"run\":" << (run_json ? run_json : "null") << "}" << std::endl;
    hlsl_probe_free_string(run_json);
    hlsl_probe_destroy(handle);
    return passed ? 0 : 2;
}

int command_compile_only(const std::string& shader_path, const std::string& out_path) {
    std::ifstream in(shader_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open shader: " + shader_path);
    }
    std::string source((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto dxil = compile_hlsl(source);
    if (!out_path.empty()) {
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(dxil.data()), static_cast<std::streamsize>(dxil.size()));
    }
    std::cout << "{\"status\":\"passed\",\"dxil_size\":" << dxil.size() << "}" << std::endl;
    return 0;
}

void usage() {
    std::cerr << "Usage:\n"
              << "  hlsl_probe --probe [--pretty-json|--compact-json]\n"
              << "  hlsl_probe --self-test\n"
              << "  hlsl_probe --linalg-test [--target cs_6_10]\n"
              << "  hlsl_probe --compile-only shader.hlsl [--out shader.dxil]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 2;
        }
        std::string command = argv[1];
        if (command == "--probe") {
            JsonOutputMode output_mode = JsonOutputMode::Pretty;
            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--pretty-json") {
                    output_mode = JsonOutputMode::Pretty;
                } else if (arg == "--compact-json") {
                    output_mode = JsonOutputMode::Compact;
                } else {
                    usage();
                    return 2;
                }
            }
            return command_probe(output_mode);
        }
        if (command == "--self-test") {
            return command_self_test();
        }
        if (command == "--linalg-test") {
            std::string target = "cs_6_10";
            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--target" && i + 1 < argc) {
                    target = argv[++i];
                }
            }
            return command_linalg_test(target);
        }
        if (command == "--compile-only") {
            if (argc < 3) {
                usage();
                return 2;
            }
            std::string out_path;
            for (int i = 3; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--out" && i + 1 < argc) {
                    out_path = argv[++i];
                }
            }
            return command_compile_only(argv[2], out_path);
        }
        usage();
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}
