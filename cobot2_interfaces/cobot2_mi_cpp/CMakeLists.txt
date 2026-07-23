cmake_minimum_required(VERSION 3.8)
project(cobot2_mi_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++17 이상 필요 (MoveGroupInterface 요구사항)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(cobot2_interfaces REQUIRED)

add_executable(cobot2_mi_node src/cobot2_mi_node.cpp)

ament_target_dependencies(cobot2_mi_node
  rclcpp
  std_msgs
  geometry_msgs
  moveit_ros_planning_interface
  moveit_msgs
  cobot2_interfaces
)

install(TARGETS
  cobot2_mi_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
