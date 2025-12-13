---
sidebar_position: 2
title: Nodes, Topics, and Services in ROS 2
---

# Nodes, Topics, and Services in ROS 2

## Understanding Nodes

Nodes are the fundamental building blocks of ROS 2 applications. They represent individual processes that perform specific functions within a robotic system.

### Node Creation and Management

#### Basic Node Structure
```cpp
#include "rclcpp/rclcpp.hpp"

class MyNode : public rclcpp::Node
{
public:
    MyNode() : Node("my_node_name")
    {
        // Node initialization code
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyNode>());
    rclcpp::shutdown();
    return 0;
}
```

#### Node Parameters
- **Dynamic configuration**: Parameters can be changed at runtime
- **Type safety**: Strong typing for parameter values
- **Namespace support**: Hierarchical parameter organization

### Node Lifecycle

ROS 2 provides a managed lifecycle system:
- **Unconfigured**: Node created but not configured
- **Inactive**: Configured but not active
- **Active**: Fully operational
- **Finalized**: Node is shutting down

## Topics - Publish/Subscribe Communication

Topics enable asynchronous, many-to-many communication in ROS 2 systems.

### Publisher Implementation

#### C++ Publisher
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class Publisher : public rclcpp::Node
{
public:
    Publisher() : Node("publisher_node")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic_name", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&Publisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello, World!";
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};
```

#### Python Publisher
```python
import rclcpp
from rclpy.node import Node
from std_msgs.msg import String

class Publisher(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(String, 'topic_name', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, World!'
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
```

### Subscriber Implementation

#### C++ Subscriber
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class Subscriber : public rclcpp::Node
{
public:
    Subscriber() : Node("subscriber_node")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic_name", 10,
            std::bind(&Subscriber::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};
```

#### Python Subscriber
```python
import rclcpp
from rclpy.node import Node
from std_msgs.msg import String

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

### Topic Design Principles

#### Message Types
- **Standard messages**: Use existing message types when possible
- **Custom messages**: Define custom message types for specific needs
- **Efficiency**: Minimize message size for real-time performance

#### Topic Naming Conventions
- **Descriptive names**: Use clear, descriptive topic names
- **Namespace organization**: Use namespaces for logical grouping
- **Consistency**: Maintain consistent naming across the system

## Services - Request/Response Communication

Services provide synchronous, one-to-one communication for request/response interactions.

### Service Implementation

#### Service Definition (.srv file)
```
# Request message
string request_data
---
# Response message
bool success
string message
```

#### Service Server
```cpp
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class ServiceServer : public rclcpp::Node
{
public:
    ServiceServer() : Node("service_server")
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
```

#### Service Client
```cpp
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class ServiceClient : public rclcpp::Node
{
public:
    ServiceClient() : Node("service_client")
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

## Advanced Communication Patterns

### Actions
Actions combine the benefits of services and topics:
- **Long-running tasks**: Handle operations that take time
- **Feedback**: Provide progress updates during execution
- **Goal preemption**: Cancel goals in progress

### Composition
- **Components**: Create nodes that can be composed into a single process
- **Performance**: Reduce inter-process communication overhead
- **Deployment flexibility**: Choose between separate processes or composition

## Physical AI Applications

### Sensor Data Distribution
- **Multi-sensor fusion**: Distribute sensor data to multiple processing nodes
- **Real-time requirements**: Use appropriate QoS settings for sensor data
- **Synchronization**: Coordinate data from multiple sensors

### Control System Integration
- **Distributed control**: Separate perception, planning, and control nodes
- **Safety systems**: Implement safety monitoring nodes
- **Human-robot interaction**: Coordinate interaction modules

## Best Practices

### Performance Considerations
- **Message frequency**: Balance update rate with computational load
- **QoS settings**: Choose appropriate settings for your application
- **Memory management**: Efficiently manage message allocation

### Error Handling
- **Connection monitoring**: Detect and handle communication failures
- **Graceful degradation**: Continue operation when possible with reduced functionality
- **Logging**: Maintain appropriate logging for debugging

## Summary

Nodes, topics, and services form the core communication infrastructure of ROS 2 systems. Understanding these concepts is essential for building distributed robotic systems that can effectively implement Physical AI principles.

In the next section, we'll explore ROS 2 packages and their role in organizing robotic software.