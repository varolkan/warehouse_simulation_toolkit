cmake_minimum_required(VERSION 3.16)
project(agv_robot)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/generate_masks.py DESTINATION lib/${PROJECT_NAME} RENAME generate_masks)
install(PROGRAMS scripts/mission_node scripts/safety_node scripts/fake_base_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY behavior_trees config launch sites urdf DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  # ROS 2 Python style. No per-file copyright headers: the licence is the repository's LICENSE.
  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_config test/test_config.py TIMEOUT 120)
  ament_add_pytest_test(test_site test/test_site.py
    ENV AGV_ROBOT_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 300)

  # The whole stack on fake_base: AMCL, Nav2, the safety gate and the mission, in real time.
  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_dry_run_delivery.py TIMEOUT 900)
  add_launch_test(test/test_dry_run_operations.py TIMEOUT 1500)
endif()

ament_package()
