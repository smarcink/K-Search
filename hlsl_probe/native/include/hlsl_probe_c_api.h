#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(HLSL_PROBE_BUILD_DLL)
#    define HLSL_PROBE_API __declspec(dllexport)
#  else
#    define HLSL_PROBE_API __declspec(dllimport)
#  endif
#else
#  define HLSL_PROBE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HlslProbeContext HlslProbeContext;
typedef HlslProbeContext* HlslProbeHandle;

typedef struct HlslProbeInputBuffer {
    const void* data;
    uint64_t size_bytes;
} HlslProbeInputBuffer;

typedef struct HlslProbeOutputBuffer {
    void* data;
    uint64_t size_bytes;
} HlslProbeOutputBuffer;

typedef struct HlslProbeRunConfig {
    uint32_t dispatch_x;
    uint32_t dispatch_y;
    uint32_t dispatch_z;
    const HlslProbeInputBuffer* inputs;
    uint32_t input_count;
    HlslProbeOutputBuffer* outputs;
    uint32_t output_count;
} HlslProbeRunConfig;

HLSL_PROBE_API HlslProbeHandle hlsl_probe_create(const char* agility_sdk_path_utf8);
HLSL_PROBE_API void hlsl_probe_destroy(HlslProbeHandle handle);

HLSL_PROBE_API const char* hlsl_probe_get_last_error(HlslProbeHandle handle);
HLSL_PROBE_API const char* hlsl_probe_get_global_last_error(void);

HLSL_PROBE_API char* hlsl_probe_get_caps_json(HlslProbeHandle handle);
HLSL_PROBE_API int hlsl_probe_run_dxil(
    HlslProbeHandle handle,
    const void* dxil_data,
    uint64_t dxil_size_bytes,
    const HlslProbeRunConfig* config,
    char** result_json);

HLSL_PROBE_API void hlsl_probe_free_string(char* value);

#ifdef __cplusplus
}
#endif
