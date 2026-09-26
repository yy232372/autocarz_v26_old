cmake_minimum_required(VERSION 3.8)
project(autocarz LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CtrlCmd.msg"
  "msg/Erp42State.msg"
  "msg/WaypointInfo.msg"
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

add_library(dbscan STATIC src/dbscan.cpp)
target_include_directories(dbscan PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

function(add_autocarz_node target_name source_file)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
  )
  ament_target_dependencies(${target_name}
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    nav_msgs
    ament_index_cpp
    pcl_conversions
  )
  target_link_libraries(${target_name}
    ${PCL_LIBRARIES}
    "${cpp_typesupport_target}"
  )
endfunction()

add_autocarz_node(roi_in src/roi_in.cpp)
add_autocarz_node(roi_out src/roi_out.cpp)
add_autocarz_node(roi_in_track src/roi_in_track.cpp)
add_autocarz_node(roi_out_track src/roi_out_track.cpp)
add_autocarz_node(ransac_in src/ransac_in.cpp)
add_autocarz_node(ransac_out src/ransac_out.cpp)
add_autocarz_node(cluster_in src/cluster_in.cpp)
add_autocarz_node(cluster_out src/cluster_out.cpp)

target_link_libraries(cluster_in dbscan)
target_link_libraries(cluster_out dbscan)

install(TARGETS
  roi_in
  roi_out
  roi_in_track
  roi_out_track
  ransac_in
  ransac_out
  cluster_in
  cluster_out
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS dbscan
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

install(PROGRAMS
  scripts/main.py
  scripts/sensor_data.py
  scripts/online.py
  scripts/ERP42RQT.py
  scripts/online_isro.py
  scripts/sensor_data_isro.py
  scripts/serialControl.py
  scripts/utilities.py
  scripts/vehicle_control.py
  scripts/path_recorder.py
  DESTINATION lib/${PROJECT_NAME}
)

# ERP42RQT.py first looks next to the installed executable.  Install the Qt
# form there as well as in share/${PROJECT_NAME}/scripts so standalone
# ros2-run execution works with both regular and symlink installs.
install(FILES
  scripts/ERP42RQT.ui
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz path msg scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(
  rosidl_default_runtime
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  ament_index_cpp
  pcl_conversions
)

ament_package()
