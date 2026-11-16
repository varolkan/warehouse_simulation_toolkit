cmake_minimum_required(VERSION 3.16)
project(agv_core)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  # ROS 2 Python style. No per-file copyright headers: the licence is the repository's LICENSE.
  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8()

  find_package(ament_cmake_pytest REQUIRED)
  file(GLOB test_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/test_*.py)
  foreach(test_file ${test_files})
    get_filename_component(test_name ${test_file} NAME_WE)
    # The closed-loop scenarios drive some 40 whole orders: a few minutes on a laptop.
    set(test_timeout 300)
    if(test_name STREQUAL "test_scenarios")
      set(test_timeout 1800)
    endif()
    ament_add_pytest_test(${test_name} ${test_file}
      PYTHON_EXECUTABLE ${Python3_EXECUTABLE}
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
      TIMEOUT ${test_timeout})
  endforeach()
endif()

ament_package()
