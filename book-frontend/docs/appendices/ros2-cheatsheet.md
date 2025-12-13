---
sidebar_position: 1
title: ROS 2 Cheatsheet
---

# ROS 2 Cheatsheet

## Core ROS 2 Commands

### Workspace Management
```bash
# Create a new workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash

# Build specific package
colcon build --packages-select my_package

# Build with additional CMake args
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Package Management
```bash
# Create a new package
ros2 pkg create --build-type ament_cmake --dependencies rclcpp std_msgs my_new_package

# List all packages
ros2 pkg list

# Show package info
ros2 pkg info my_package

# Find package path
ros2 pkg prefix my_package
```

### Node Management
```bash
# List active nodes
ros2 node list

# Show node info
ros2 node info /my_node

# Run a node
ros2 run my_package my_node

# Run with arguments
ros2 run my_package my_node --arg1 value1 --arg2 value2
```

### Topic Management
```bash
# List topics
ros2 topic list

# Show topic info
ros2 topic info /my_topic

# Echo topic data
ros2 topic echo /my_topic

# Echo with specific message count
ros2 topic echo /my_topic --field data 10

# Publish to topic
ros2 topic pub /my_topic std_msgs/String "data: 'Hello World'"

# Show topic type
ros2 topic type /my_topic

# Find topics by type
ros2 topic list -t
```

### Service Management
```bash
# List services
ros2 service list

# Show service info
ros2 service info /my_service

# Call a service
ros2 service call /my_service example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"

# Show service type
ros2 service type /my_service
```

### Action Management
```bash
# List actions
ros2 action list

# Show action info
ros2 action info /my_action

# Send goal to action
ros2 action send_goal /my_action example_interfaces/action/Fibonacci "{order: 5}"
```

### Parameter Management
```bash
# List parameters of a node
ros2 param list /my_node

# Get parameter value
ros2 param get /my_node param_name

# Set parameter value
ros2 param set /my_node param_name param_value

# Load parameters from file
ros2 param load /my_node /path/to/params.yaml
```

### Launch Files
```bash
# Run a launch file
ros2 launch my_package my_launch_file.py

# Run with arguments
ros2 launch my_package my_launch_file.py arg_name:=arg_value

# Run with multiple arguments
ros2 launch my_package my_launch_file.py param1:=value1 param2:=value2
```

## Common Message Types

### Standard Messages
```cpp
// String message
std_msgs::msg::String msg;
msg.data = "Hello World";

// Integer messages
std_msgs::msg::Int32 int_msg;
int_msg.data = 42;

std_msgs::msg::Float64 float_msg;
float_msg.data = 3.14;

// Boolean message
std_msgs::msg::Bool bool_msg;
bool_msg.data = true;

// Header message
std_msgs::msg::Header header;
header.stamp = this->get_clock()->now();
header.frame_id = "base_link";
```

### Geometry Messages
```cpp
// Point
geometry_msgs::msg::Point point;
point.x = 1.0;
point.y = 2.0;
point.z = 3.0;

// Pose
geometry_msgs::msg::Pose pose;
pose.position = point;
pose.orientation.w = 1.0;  // No rotation

// Twist (velocity)
geometry_msgs::msg::Twist twist;
twist.linear.x = 0.5;   // Move forward at 0.5 m/s
twist.angular.z = 0.2;  // Rotate at 0.2 rad/s
```

### Sensor Messages
```cpp
// Laser scan
sensor_msgs::msg::LaserScan scan;
scan.angle_min = -M_PI/2;
scan.angle_max = M_PI/2;
scan.angle_increment = M_PI/180;  // 1 degree increments
scan.range_min = 0.1;
scan.range_max = 10.0;
scan.ranges = std::vector<float>(181, 5.0);  // 181 readings at 5m

// Image
sensor_msgs::msg::Image image;
image.height = 480;
image.width = 640;
image.encoding = "rgb8";
image.is_bigendian = false;
image.step = 640 * 3;  // width * bytes per pixel
image.data = std::vector<uint8_t>(640 * 480 * 3, 0);
```

## Creating a Basic ROS 2 Node

### C++ Node Template
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello, world! " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}
```

### Python Node Template
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### CMakeLists.txt for C++ Package
```cmake
cmake_minimum_required(VERSION 3.8)
project(my_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Create executable
add_executable(talker src/talker.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

# Install targets
install(TARGETS
  talker
  DESTINATION lib/${PROJECT_NAME}
)

# Package configuration
ament_package()
```

### package.xml
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_package</name>
  <version>0.0.0</version>
  <description>My ROS 2 Package</description>
  <maintainer email="user@todo.todo">User</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Launch Files

### Python Launch File
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='talker',
            name='my_talker',
            parameters=[
                {'param1': 'value1'},
                {'param2': 123}
            ],
            remappings=[
                ('original_topic', 'remapped_topic')
            ]
        ),
        Node(
            package='my_package',
            executable='listener',
            name='my_listener'
        )
    ])
```

### YAML Parameter File
```yaml
my_node:
  ros__parameters:
    param1: value1
    param2: 123
    nested:
      param3: 456
    array_param: [1, 2, 3, 4, 5]
```

## Common Design Patterns

### Publisher-Subscriber Pattern
```cpp
class DataProcessor : public rclcpp::Node
{
public:
    DataProcessor() : Node("data_processor")
    {
        // Create subscriber
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "input_topic", 10,
            std::bind(&DataProcessor::topic_callback, this, std::placeholders::_1));

        // Create publisher
        publisher_ = this->create_publisher<std_msgs::msg::String>("output_topic", 10);
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());

        // Process data
        auto processed_msg = std_msgs::msg::String();
        processed_msg.data = "Processed: " + msg->data;

        // Publish result
        publisher_->publish(processed_msg);
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};
```

### Client-Server Pattern
```cpp
// Service Server
class AddTwoIntsService : public rclcpp::Node
{
public:
    AddTwoIntsService() : Node("add_two_ints_server")
    {
        service_ = this->create_service<example_interfaces::srv::AddTwoInts>(
            "add_two_ints",
            [this](const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
                   example_interfaces::srv::AddTwoInts::Response::SharedPtr response) {
                response->sum = request->a + request->b;
                RCLCPP_INFO(this->get_logger(), "Incoming request: %ld + %ld = %ld",
                           request->a, request->b, response->sum);
            });
    }

private:
    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
};

// Service Client
class AddTwoIntsClient : public rclcpp::Node
{
public:
    AddTwoIntsClient() : Node("add_two_ints_client")
    {
        client_ = this->create_client<example_interfaces::srv::AddTwoInts>("add_two_ints");
        while (!client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for service");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
        }
        send_request();
    }

private:
    void send_request()
    {
        auto request = std::make_shared<example_interfaces::srv::AddTwoInts::Request>();
        request->a = 2;
        request->b = 3;

        auto result_future = client_->async_send_request(request);
        // Handle result when available
    }

    rclcpp::Client<example_interfaces::srv::AddTwoInts>::SharedPtr client_;
};
```

## Debugging and Monitoring

### Useful Debugging Commands
```bash
# Monitor topic with specific frequency
ros2 topic echo /my_topic --rate 10

# Monitor topic with field filtering
ros2 topic echo /my_topic --field data

# Monitor multiple topics
toplevel=$(ros2 topic list | grep -E "(topic1|topic2|topic3)")
for topic in $toplevel; do
    ros2 topic echo $topic &
done

# Check network usage
ros2 topic hz /my_topic

# Monitor node resource usage
top  # Then filter for your node's process
```

### Logging
```cpp
// Different log levels
RCLCPP_DEBUG(this->get_logger(), "Debug message: %s", variable.c_str());
RCLCPP_INFO(this->get_logger(), "Info message: %d", number);
RCLCPP_WARN(this->get_logger(), "Warning message");
RCLCPP_ERROR(this->get_logger(), "Error message: %s", error_str.c_str());
RCLCPP_FATAL(this->get_logger(), "Fatal error message");
```

### Testing
```cpp
// Create a simple test
#include <gtest/gtest.h>
#include "my_package/my_node.hpp"

TEST(MyNodeTest, BasicFunctionality) {
    // Create node instance
    auto node = std::make_shared<MyNode>();

    // Test specific functionality
    EXPECT_TRUE(node->is_initialized());
}
```

## Performance Optimization

### Efficient Message Handling
```cpp
// Use shared_ptr to avoid copying large messages
void callback(const std::shared_ptr<const MyMessage> msg)
{
    // Process message without copying
}

// Use intraprocess communication when possible
// In launch file configuration:
# In your launch file, you can enable intraprocess communication
# for nodes that are in the same process:
# <node name="node1" pkg="pkg" exec="exec"
#       parameters="{'use_intraprocess_comms': True}">
```

### Quality of Service (QoS) Settings
```cpp
// Common QoS profiles
rclcpp::QoS qos_profile(10);  // Keep last 10 messages

// Reliable communication
qos_profile.reliable();

// Best effort (faster, but may lose messages)
qos_profile.best_effort();

// Keep all messages (memory intensive)
qos_profile.keep_all();

// Durability settings
qos_profile.transient_local();  // Keep messages for late-joining subscribers
qos_profile.durability_volatile();  // Don't keep messages for late joiners

// Example publisher with custom QoS
auto publisher = this->create_publisher<MyMessage>(
    "topic_name",
    rclcpp::QoS(10).reliable().transient_local());
```

### Lifecycle Nodes
```cpp
#include "rclcpp_lifecycle/lifecycle_node.hpp"

class LifecycleNodeExample : public rclcpp_lifecycle::LifecycleNode
{
public:
    LifecycleNodeExample() : rclcpp_lifecycle::LifecycleNode("lifecycle_node")
    {
    }

private:
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &)
    {
        // Initialize resources
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &)
    {
        // Activate functionality
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &)
    {
        // Deactivate functionality
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State &)
    {
        // Clean up resources
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
};
```

## Common Issues and Solutions

### Topic Connection Issues
```bash
# Check if nodes can see each other
ros2 topic list
ros2 node list

# Verify topic type matches
ros2 topic type /topic_name
ros2 interface show MessageType

# Check for message serialization issues
ros2 topic echo /topic_name --field field_name
```

### Performance Issues
```bash
# Monitor node performance
htop  # Look for CPU usage
ros2 run demo_nodes_cpp listener  # Check message rate

# Reduce message frequency
# In your node: increase timer period or add rate limiting

# Use smaller message types when possible
# Consider using compressed images for large data
```

### Memory Management
```cpp
// Use smart pointers to avoid memory leaks
std::shared_ptr<MyMessage> msg = std::make_shared<MyMessage>();

// Avoid copying large messages unnecessarily
void callback(const std::shared_ptr<const MyMessage> msg)  // Pass by const reference
{
    // Process without copying
}
```

## Advanced Features

### Composition (Multiple nodes in one process)
```cpp
// In launch file:
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='my_package',
                plugin='MyPackage::Node1',
                name='node1'),
            ComposableNode(
                package='my_package',
                plugin='MyPackage::Node2',
                name='node2'),
        ],
        output='screen',
    )

    return LaunchDescription([container])
```

### TF2 (Transform Library)
```cpp
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

class TransformNode : public rclcpp::Node
{
public:
    TransformNode() : Node("transform_node")
    {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

    void lookup_transform()
    {
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform(
                "target_frame", "source_frame",
                tf2::TimePointZero);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform: %s", ex.what());
        }
    }

private:
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};
```

This cheatsheet provides a comprehensive reference for ROS 2 development, covering the most commonly used commands, patterns, and best practices needed for developing robotic applications.