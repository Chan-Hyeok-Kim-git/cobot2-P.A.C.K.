
cmake_minimum_required(VERSION 3.8)
project(cobot2_interfaces)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# ── 의존성 ─────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# ── ★ 핵심: 메시지 파일 등록 (이 부분이 빠져있었음) ──────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ValidatedGrasp.msg"
  DEPENDENCIES
    std_msgs
    geometry_msgs
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
ament_package()

