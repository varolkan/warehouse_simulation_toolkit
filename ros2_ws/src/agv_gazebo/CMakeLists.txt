cmake_minimum_required(VERSION 3.16)
project(agv_gazebo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/generate_world.py DESTINATION lib/${PROJECT_NAME} RENAME generate_world)
install(DIRECTORY config models worlds DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8()
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_world test/test_world.py
    ENV AGV_GAZEBO_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 120)
endif()

ament_package()
