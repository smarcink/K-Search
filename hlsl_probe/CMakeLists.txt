cmake_minimum_required(VERSION 3.25)

project(hlsl_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)

set(DXC_ROOT "${REPO_ROOT}/thirdparty/dxc_preview_2026_04_22" CACHE PATH "DirectX Shader Compiler package root")
set(AGILITY_ROOT "${REPO_ROOT}/thirdparty/microsoft.direct3d.d3d12.1.720.0-preview" CACHE PATH "DirectX 12 Agility SDK package root")
set(AGILITY_SDK_VERSION "720" CACHE STRING "DirectX 12 Agility SDK version passed to D3D12SDKVersion/CreateDeviceFactory")

set(DXC_EXE "${DXC_ROOT}/bin/x64/dxc.exe")
set(DXC_HLSL_INCLUDE "${DXC_ROOT}/inc/hlsl")
set(AGILITY_INCLUDE "${AGILITY_ROOT}/build/native/include")
set(AGILITY_D3DX12_INCLUDE "${AGILITY_ROOT}/build/native/include/d3dx12")
set(AGILITY_BIN_X64 "${AGILITY_ROOT}/build/native/bin/x64")

foreach(required_path IN ITEMS "${DXC_EXE}" "${DXC_HLSL_INCLUDE}" "${AGILITY_INCLUDE}" "${AGILITY_BIN_X64}/D3D12Core.dll")
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR "Required HLSL probe dependency not found: ${required_path}")
    endif()
endforeach()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/native/src/hlsl_probe_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/hlsl_probe_config.h"
    @ONLY
)

add_library(hlsl_probe_native SHARED
    native/src/hlsl_probe.cpp
)

target_include_directories(hlsl_probe_native
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/native/include"
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${AGILITY_INCLUDE}"
        "${AGILITY_D3DX12_INCLUDE}"
)

target_compile_definitions(hlsl_probe_native PRIVATE
    HLSL_PROBE_BUILD_DLL
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

target_compile_options(hlsl_probe_native PRIVATE /W4 /permissive- /EHsc)
target_link_libraries(hlsl_probe_native PRIVATE d3d12 dxgi dxguid)

add_executable(hlsl_probe_cli
    native/src/main.cpp
)

set_target_properties(hlsl_probe_cli PROPERTIES OUTPUT_NAME hlsl_probe)

target_include_directories(hlsl_probe_cli PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/native/include"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${AGILITY_INCLUDE}"
)

target_compile_definitions(hlsl_probe_cli PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

target_compile_options(hlsl_probe_cli PRIVATE /W4 /permissive- /EHsc)
target_link_libraries(hlsl_probe_cli PRIVATE hlsl_probe_native)

function(copy_agility_runtime target_name)
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target_name}>/D3D12"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_BIN_X64}/D3D12Core.dll" "$<TARGET_FILE_DIR:${target_name}>/D3D12/D3D12Core.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_BIN_X64}/d3d12SDKLayers.dll" "$<TARGET_FILE_DIR:${target_name}>/D3D12/d3d12SDKLayers.dll"
        VERBATIM
    )
endfunction()

copy_agility_runtime(hlsl_probe_native)
copy_agility_runtime(hlsl_probe_cli)
