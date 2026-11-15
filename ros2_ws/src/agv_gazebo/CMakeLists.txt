cmake_minimum_required(VERSION 3.16)
project(agv_gazebo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/generate_world.py DESTINATION lib/${PROJECT_NAME} RENAME generate_world)
install(DIRECTORY config models worlds DESTINATION share/${PROJECT_NAME})

ament_package()
