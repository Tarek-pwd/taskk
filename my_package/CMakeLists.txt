cmake_minimum_required(VERSION 3.0.2)
project(my_package)

## Compile as C++11
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)

## Declare ROS messages and services
add_service_files(
  FILES
  KalmanFilter.srv
)

## Generate messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare and build the C++ Kalman filter
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES kalman_filter
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library for the Kalman filter
add_library(kalman_filter
  src/kalman_filter.cpp
)

## Link the Kalman filter library with the catkin and OpenCV libraries
target_link_libraries(kalman_filter
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## Declare a C++ executable for the ROS service node
add_executable(kalman_node src/kalman_node.cpp)

## Link the C++ executable with the Kalman filter library and other dependencies
target_link_libraries(kalman_node
  kalman_filter
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## Install targets
install(TARGETS kalman_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS kalman_filter
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

## Mark header files for installation
install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
