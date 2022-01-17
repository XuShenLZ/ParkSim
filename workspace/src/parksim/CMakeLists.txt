cmake_minimum_required(VERSION 3.5)
project(parksim)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ParametricPoseMsg.msg"
  "msg/ParametricVelocityMsg.msg"
  "msg/BodyAngularAccelerationMsg.msg"
  "msg/BodyAngularVelocityMsg.msg"
  "msg/BodyLinearAccelerationMsg.msg"
  "msg/BodyLinearVelocityMsg.msg"
  "msg/OrientationEulerMsg.msg"
  "msg/OrientationQuaternionMsg.msg"
  "msg/PositionMsg.msg"
  "msg/VehicleStateMsg.msg"
  "msg/EncoderMsg.msg"
  "msg/PredictionMsg.msg"
  "msg/VehicleActuationMsg.msg"
  DEPENDENCIES builtin_interfaces std_msgs
  )
ament_export_dependencies(rosidl_default_runtime)

install(PROGRAMS
  src/visualizer_node.py
  src/test_vehicle_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR ../../../python/${PROJECT_NAME}
)

ament_package()
