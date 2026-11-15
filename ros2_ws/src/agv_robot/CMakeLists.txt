cmake_minimum_required(VERSION 3.16)
project(agv_robot)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/generate_masks.py DESTINATION lib/${PROJECT_NAME} RENAME generate_masks)
install(DIRECTORY behavior_trees config launch sites urdf DESTINATION share/${PROJECT_NAME})

ament_package()
