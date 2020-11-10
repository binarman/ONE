file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(luci_run ${SOURCES})
target_include_directories(luci_run PRIVATE include)
target_include_directories(luci_run PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(luci_run arser)
target_link_libraries(luci_run luci_import)
target_link_libraries(luci_run luci_interpreter)
target_link_libraries(luci_run vconone)

install(TARGETS luci_run DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

#nnas_find_package(GTest REQUIRED)
#GTest_AddTest(luci_run_function_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/RecordFunction.test.cpp")
#target_include_directories(luci_run_function_test PRIVATE include)
