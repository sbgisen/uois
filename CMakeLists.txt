cmake_minimum_required(VERSION 3.0.2)
project(uois)

find_package(
  catkin REQUIRED
  COMPONENTS sensor_msgs
             cv_bridge
             rospy
             message_generation
)

add_service_files(FILES Inference.srv)

generate_messages(DEPENDENCIES sensor_msgs)

catkin_package(
  CATKIN_DEPENDS
    sensor_msgs
    cv_bridge
    rospy
    message_runtime
)
