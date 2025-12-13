---
sidebar_position: 3
title: ROS 2 Packages and Build System
---

# ROS 2 Packages and Build System

## Understanding ROS 2 Packages

ROS 2 packages are the fundamental units of software organization in ROS 2. They contain source code, configuration files, launch files, and other resources needed to build and run ROS 2 applications.

### Package Structure

A typical ROS 2 package follows this structure:
```
my_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── src/                    # Source code files
├── include/                # Header files (C++)
├── launch/                 # Launch files
├── config/                 # Configuration files
├── test/                   # Test files
├── scripts/                # Script files
└── README.md               # Documentation
```

### Package Metadata (package.xml)

The `package.xml` file contains essential metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_package</name>
  <version>0.0.0</version>
  <description>My ROS 2 Package</description>
  <maintainer email="user@example.com">User Name</maintainer>
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

## Build System (Colcon)

Colcon is the build system used in ROS 2, replacing catkin from ROS 1.

### Colcon Commands

#### Building Packages
```bash
# Build all packages in the workspace
colcon build

# Build specific package
colcon build --packages-select my_package

# Build with parallel jobs
colcon build --parallel-workers 4

# Build with additional CMake arguments
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

#### Running Tests
```bash
# Run all tests
colcon test

# Run tests for specific package
colcon test --packages-select my_package

# View test results
colcon test-result --all
```

## CMake Configuration

### Basic CMakeLists.txt

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
add_executable(my_node src/my_node.cpp)
ament_target_dependencies(my_node
  rclcpp
  std_msgs
)

# Install targets
install(TARGETS
  my_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Package configuration
ament_package()
```

## Package Development Workflow

### Creating a New Package

#### C++ Package
```bash
ros2 pkg create --build-type ament_cmake --dependencies rclcpp std_msgs my_cpp_package
```

#### Python Package
```bash
ros2 pkg create --build-type ament_python --dependencies rclpy std_msgs my_python_package
```

### Package Dependencies

#### Build Dependencies
- Required during compilation
- Listed in `package.xml` as `<build_depend>` or `<buildtool_depend>`

#### Execution Dependencies
- Required at runtime
- Listed in `package.xml` as `<exec_depend>`

#### Test Dependencies
- Required for testing
- Listed in `package.xml` as `<test_depend>`

## Advanced Package Features

### Launch Files

Launch files define how to start multiple nodes with specific configurations:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='my_node',
            name='my_node',
            parameters=[
                {'param1': 'value1'},
                {'param2': 123}
            ],
            remappings=[
                ('original_topic', 'remapped_topic')
            ]
        )
    ])
```

### Parameter Files

YAML files for parameter configuration:

```yaml
my_node:
  ros__parameters:
    param1: value1
    param2: 123
    nested:
      param3: 456
```

### Message and Service Definitions

#### Custom Messages (.msg)
```
# Custom message definition
string name
int32 id
float64[] values
geometry_msgs/Pose pose
```

#### Custom Services (.srv)
```
# Request
string command
---
# Response
bool success
string message
```

## Physical AI Package Organization

### Modular Architecture
- **Perception packages**: Handle sensor data processing
- **Control packages**: Implement control algorithms
- **Planning packages**: Path planning and navigation
- **Interface packages**: Human-robot interaction

### Best Practices for Physical AI

#### Code Organization
- **Clear interfaces**: Well-defined APIs between packages
- **Reusability**: Design packages for reuse across different robots
- **Documentation**: Comprehensive documentation for each package

#### Performance Considerations
- **Efficient builds**: Minimize build times with proper dependency management
- **Memory usage**: Optimize for embedded systems with limited resources
- **Real-time constraints**: Consider timing requirements in package design

## Testing and Quality Assurance

### Unit Testing
- **gtest**: C++ unit testing framework
- **pytest**: Python unit testing framework
- **Mock objects**: Test components in isolation

### Integration Testing
- **System-level tests**: Test complete robot functionality
- **Simulation integration**: Test in simulated environments
- **Hardware-in-the-loop**: Test with real hardware when possible

### Code Quality Tools
- **Linters**: Static code analysis
- **Formatters**: Code formatting consistency
- **Coverage**: Test coverage measurement

## Deployment and Distribution

### Package Distribution
- **Debian packages**: Binary distribution for Ubuntu
- **Docker containers**: Containerized deployment
- **Snap packages**: Universal Linux packages

### Version Management
- **Semantic versioning**: Clear version numbering scheme
- **Release procedures**: Proper release and tagging
- **Backwards compatibility**: Maintain API compatibility

## Common ROS 2 Packages for Physical AI

### Navigation Stack
- **navigation2**: Path planning and navigation
- **slam_toolbox**: Simultaneous localization and mapping

### Perception Packages
- **vision_opencv**: OpenCV integration
- **image_transport**: Efficient image handling
- **pointcloud**: 3D point cloud processing

### Control Packages
- **ros_controllers**: Robot controller framework
- **realtime_tools**: Real-time safe utilities

## Summary

ROS 2 packages provide the essential organization and build infrastructure for robotic applications. Proper package design and organization are crucial for developing maintainable and scalable Physical AI systems.

In the next module, we'll explore simulation environments for developing and testing Physical AI systems.