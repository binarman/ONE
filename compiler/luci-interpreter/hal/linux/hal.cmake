macro(initialize_hal)
    nnas_find_package(TensorFlowSource EXACT 2.3.0 QUIET)
    nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.3.0 QUIET)
    nnas_find_package(TensorFlowEigenSource EXACT 2.3.0 QUIET)
    nnas_find_package(TensorFlowRuySource EXACT 2.3.0 QUIET)

    if (NOT TensorFlowSource_FOUND)
        message(STATUS "Skipping luci-interpreter: TensorFlow not found")
        return()
    endif ()

    if (NOT TensorFlowGEMMLowpSource_FOUND)
        message(STATUS "Skipping luci-interpreter: gemmlowp not found")
        return()
    endif ()

    if (NOT TensorFlowEigenSource_FOUND)
        message(STATUS "Skipping luci-interpreter: Eigen not found")
        return()
    endif ()

    if (NOT TensorFlowRuySource_FOUND)
        message(STATUS "Skipping luci-interpreter: Ruy not found")
        return()
    endif ()

    find_package(Threads REQUIRED)

    set(HAL_INITIALIZED TRUE)
endmacro()

macro(add_hal_to_target TGT)
    target_include_directories(${TGT} PRIVATE "${HAL}")
    target_include_directories(${TGT} SYSTEM PRIVATE
            "${TensorFlowRuySource_DIR}"
            "${TensorFlowGEMMLowpSource_DIR}"
            "${TensorFlowEigenSource_DIR}"
            "${TensorFlowSource_DIR}")
    target_include_directories(${TGT} PRIVATE ${LUCI_INTERPRETER_HAL_DIR})
    target_link_libraries(${TGT} PRIVATE Threads::Threads)
endmacro()
