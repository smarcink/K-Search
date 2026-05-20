#include "hlsl_probe_c_api.h"
#include "hlsl_probe_config.h"

#include <windows.h>
#include <d3d12.h>

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

    std::ostringstream cmd;
    cmd << quote_arg(HLSL_PROBE_DEFAULT_DXC_PATH)
        << " -T " << target
        << " -E main"
        << " -HV 2021"
        << " -enable-16bit-types"
        << " -I " << quote_arg(HLSL_PROBE_DEFAULT_HLSL_INCLUDE)
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

void print_and_free(char* text) {
    if (text) {
        std::cout << text << std::endl;
        hlsl_probe_free_string(text);
    }
}

int command_probe() {
    HlslProbeHandle handle = create_probe_or_throw();
    char* caps = hlsl_probe_get_caps_json(handle);
    if (!caps) {
        std::cerr << hlsl_probe_get_last_error(handle) << std::endl;
        hlsl_probe_destroy(handle);
        return 1;
    }
    print_and_free(caps);
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
    HlslProbeOutputBuffer output = { &value, sizeof(value) };
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
              << "  hlsl_probe --probe\n"
              << "  hlsl_probe --self-test\n"
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
            return command_probe();
        }
        if (command == "--self-test") {
            return command_self_test();
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
