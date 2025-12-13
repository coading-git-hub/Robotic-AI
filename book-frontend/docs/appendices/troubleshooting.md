---
sidebar_position: 3
title: Troubleshooting Guide
---

# Troubleshooting Guide

## Common ROS 2 Issues

### Build Issues

#### Problem: `colcon build` fails with "package not found"
**Symptoms**:
```
Starting >>> my_package
--- stderr: my_package
CMake Error at CMakeLists.txt:10 (find_package):
  By not providing "Findament_cmake.cmake" in CMake's module path...
```

**Solutions**:
1. Source the ROS 2 setup script:
   ```bash
   source /opt/ros/humble/setup.bash  # or your ROS 2 distro
   ```

2. Check if dependencies are installed:
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. Verify package.xml dependencies are correct:
   ```xml
   <depend>rclcpp</depend>
   <depend>std_msgs</depend>
   ```

#### Problem: "command not found" after sourcing ROS 2
**Symptoms**: Commands like `ros2 run` or `ros2 topic` not found

**Solutions**:
1. Check if ROS 2 is properly installed:
   ```bash
   echo $ROS_DISTRO
   ```

2. Re-source the setup script:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

3. Add to your `.bashrc` or `.zshrc`:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

### Node Communication Issues

#### Problem: Topics not connecting between nodes
**Symptoms**: Nodes can't communicate despite running

**Solutions**:
1. Check if nodes are running:
   ```bash
   ros2 node list
   ```

2. Verify topic names match exactly:
   ```bash
   ros2 topic list
   ros2 topic info /topic_name
   ```

3. Check topic types match:
   ```bash
   ros2 topic type /publisher_topic
   ros2 topic type /subscriber_topic
   ```

4. Verify QoS profiles are compatible:
   ```cpp
   // In publisher
   auto publisher = this->create_publisher<MessageType>(
       "topic_name",
       rclcpp::QoS(10).reliable());  // Match subscriber QoS
   ```

#### Problem: Service calls failing
**Symptoms**: `ros2 service call` returns "service not available"

**Solutions**:
1. Check if service server is running:
   ```bash
   ros2 service list
   ```

2. Verify service type:
   ```bash
   ros2 service type /service_name
   ```

3. Add service availability check in client code:
   ```cpp
   while (!client->wait_for_service(std::chrono::seconds(1))) {
       if (!rclcpp::ok()) {
           RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for service");
           return;
       }
       RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
   }
   ```

### Performance Issues

#### Problem: High CPU usage
**Symptoms**: Node consuming excessive CPU resources

**Solutions**:
1. Check timer rates:
   ```cpp
   // Avoid very high frequency timers
   timer_ = this->create_wall_timer(  // Too fast!
       1ms, std::bind(&MyNode::callback, this));  // Use appropriate rate
   ```

2. Add rate limiting:
   ```cpp
   #include <rclcpp/rate.hpp>
   rclcpp::Rate rate(10);  // 10 Hz
   while(rclcpp::ok()) {
       // Your code here
       rate.sleep();
   }
   ```

3. Optimize callbacks:
   ```cpp
   // Avoid heavy processing in callbacks
   void callback(const Message::SharedPtr msg) {
       // Just store data, process in separate thread
       std::lock_guard<std::mutex> lock(mutex_);
       latest_message_ = msg;
   }
   ```

#### Problem: Memory leaks
**Symptoms**: Process memory usage growing over time

**Solutions**:
1. Use smart pointers:
   ```cpp
   // Good
   auto msg = std::make_shared<MessageType>();

   // Avoid raw pointers when possible
   // MessageType* msg = new MessageType();  // Remember to delete!
   ```

2. Check for circular references with shared_ptr:
   ```cpp
   // Use weak_ptr to break cycles
   std::shared_ptr<Node> node = std::make_shared<Node>();
   std::weak_ptr<Node> weak_node = node;  // Breaks the cycle
   ```

## Gazebo Simulation Issues

### Physics Instability

#### Problem: Robot shaking or vibrating
**Symptoms**: Robot joints oscillating rapidly or objects jittering

**Solutions**:
1. Increase solver iterations in world file:
   ```xml
   <physics type='ode'>
     <ode>
       <solver>
         <iters>100</iters>  <!-- Increase from default -->
         <sor>1.3</sor>
       </solver>
     </ode>
   </physics>
   ```

2. Reduce time step:
   ```xml
   <physics type='ode'>
     <max_step_size>0.0005</max_step_size>  <!-- Smaller time step -->
     <real_time_update_rate>2000</real_time_update_rate>
   </physics>
   ```

3. Check mass properties:
   ```xml
   <link name='link_name'>
     <inertial>
       <mass>1.0</mass>
       <inertia>
         <ixx>0.01</ixx>
         <ixy>0</ixy>
         <ixz>0</ixz>
         <iyy>0.01</iyy>
         <iyz>0</iyz>
         <izz>0.01</izz>
       </inertia>
     </inertial>
   </link>
   ```

#### Problem: Objects passing through each other
**Symptoms**: Collision detection not working properly

**Solutions**:
1. Check collision geometry:
   ```xml
   <collision name='collision'>
     <geometry>
       <box>
         <size>0.5 0.5 0.5</size>  <!-- Ensure proper size -->
       </box>
     </geometry>
   </collision>
   ```

2. Adjust contact parameters:
   ```xml
   <physics type='ode'>
     <ode>
       <constraints>
         <contact_surface_layer>0.005</contact_surface_layer>  <!-- Increase from default -->
         <contact_max_correcting_vel>100</contact_max_correcting_vel>
       </constraints>
     </ode>
   </physics>
   ```

3. Verify mass ratios:
   ```xml
   <!-- Heavy objects should be significantly heavier than light objects -->
   <link name='ground'>
     <inertial>
       <mass>1000.0</mass>  <!-- Very heavy ground -->
     </inertial>
   </link>
   ```

### Rendering Issues

#### Problem: Slow rendering or low FPS
**Symptoms**: Gazebo running slowly or with poor visual quality

**Solutions**:
1. Disable unnecessary visual elements:
   ```bash
   gzserver --verbose  # Run without GUI for better performance
   ```

2. Reduce visual complexity:
   ```xml
   <visual name='visual'>
     <geometry>
       <box><size>1 1 1</size></box>  <!-- Use simple geometry -->
     </geometry>
     <!-- Remove complex materials if not needed -->
   </visual>
   ```

3. Adjust rendering settings:
   ```bash
   # Set environment variables
   export GAZEBO_RENDER_STAGES=0  # Reduce rendering complexity
   export OGRE_RESOURCE_PATH=/usr/lib/x86_64-linux-gnu/OGRE-1.9.0
   ```

## Unity Simulation Issues

### Physics Problems

#### Problem: Robot falling through ground
**Symptoms**: Robot or objects fall through surfaces

**Solutions**:
1. Check ground plane setup:
   ```csharp
   // Ensure ground has proper collider and is static
   public class GroundSetup : MonoBehaviour
   {
       void Start()
       {
           // Add collider if missing
           if (GetComponent<Collider>() == null)
           {
               gameObject.AddComponent<BoxCollider>();
           }

           // Make static for better performance
           gameObject.isStatic = true;
       }
   }
   ```

2. Verify physics material:
   ```csharp
   public PhysicMaterial groundMaterial;

   void Start()
   {
       Collider groundCollider = GetComponent<Collider>();
       groundCollider.material = groundMaterial;
       groundMaterial.staticFriction = 1.0f;
       groundMaterial.dynamicFriction = 1.0f;
   }
   ```

3. Check robot's Rigidbody settings:
   ```csharp
   public class RobotSetup : MonoBehaviour
   {
       void Start()
       {
           Rigidbody rb = GetComponent<Rigidbody>();
           rb.useGravity = true;
           rb.interpolation = RigidbodyInterpolation.Interpolate;  // Smoother movement
       }
   }
   ```

#### Problem: Jittery joint movement
**Symptoms**: Robot joints moving erratically or shaking

**Solutions**:
1. Adjust solver settings:
   ```csharp
   public class PhysicsSetup : MonoBehaviour
   {
       void Start()
       {
           Physics.defaultSolverIterations = 10;        // Increase from default
           Physics.defaultSolverVelocityIterations = 8; // Increase from default
       }
   }
   ```

2. Use ConfigurableJoint for precise control:
   ```csharp
   public class RobotJoint : MonoBehaviour
   {
       public ConfigurableJoint joint;

       void Start()
       {
           joint = GetComponent<ConfigurableJoint>();

           // Set drive for position control
           JointDrive drive = new JointDrive();
           drive.positionSpring = 10000f;   // High stiffness
           drive.positionDamper = 1000f;    // Appropriate damping
           drive.maximumForce = 100f;

           joint.slerpDrive = drive;
       }
   }
   ```

### Performance Issues

#### Problem: Low frame rate in simulation
**Symptoms**: Unity simulation running slowly

**Solutions**:
1. Reduce visual quality:
   ```csharp
   void Start()
   {
       // Reduce rendering quality
       QualitySettings.SetQualityLevel(0);  // Fastest

       // Disable unnecessary effects
       RenderSettings.fog = false;
       DynamicGI.enabled = false;
   }
   ```

2. Optimize mesh complexity:
   ```csharp
   // Use LOD system
   public LODGroup lodGroup;
   public LOD[] lods;

   void Start()
   {
       lodGroup = GetComponent<LODGroup>();
       lodGroup.SetLODS(lods);
   }
   ```

3. Use object pooling:
   ```csharp
   public class ObjectPool : MonoBehaviour
   {
       public GameObject prefab;
       public int poolSize = 10;
       private Queue<GameObject> objectPool;

       void Start()
       {
           objectPool = new Queue<GameObject>();
           for (int i = 0; i < poolSize; i++)
           {
               GameObject obj = Instantiate(prefab);
               obj.SetActive(false);
               objectPool.Enqueue(obj);
           }
       }

       public GameObject GetObject()
       {
           GameObject obj = objectPool.Dequeue();
           obj.SetActive(true);
           objectPool.Enqueue(obj);
           return obj;
       }
   }
   ```

## NVIDIA Isaac Sim Issues

### USD Model Issues

#### Problem: Model not loading correctly
**Symptoms**: Robot model appears deformed or doesn't load

**Solutions**:
1. Validate USD file:
   ```bash
   usdview your_model.usd  # Check if model displays correctly
   usdzconvert your_model.usd  # Validate USD structure
   ```

2. Check stage units:
   ```python
   # In your Python script
   world = World(stage_units_in_meters=1.0)  # Ensure consistent units
   ```

3. Verify joint setup:
   ```usda
   def PhysicsJoint "joint1"
   {
       rel physics:body0 = </Robot/base_link>
       rel physics:body1 = </Robot/arm_link>
       # Ensure proper body references
   }
   ```

### Python API Issues

#### Problem: Robot not responding to control commands
**Symptoms**: Robot joints don't move despite applying actions

**Solutions**:
1. Check controller initialization:
   ```python
   def setup_post_load(self):
       # Ensure controllers are properly initialized
       self._articulation_controller = self.get_articulation_controller()
   ```

2. Verify joint names match:
   ```python
   # Check joint names
   joint_names = robot.get_joint_names()
   print("Available joints:", joint_names)
   ```

3. Apply actions correctly:
   ```python
   # Use proper action format
   from omni.isaac.core.articulations import ArticulationAction

   # For position control
   position_action = ArticulationAction(joint_positions=desired_positions)
   robot.get_articulation_controller().apply_action(position_action)
   ```

## Common Hardware Interface Issues

### Sensor Integration Problems

#### Problem: Sensor data not updating
**Symptoms**: Sensor topics show old or no data

**Solutions**:
1. Check sensor configuration:
   ```xml
   <sensor name='camera' type='camera'>
     <always_on>1</always_on>  <!-- Ensure sensor is always on -->
     <update_rate>30</update_rate>
     <visualize>true</visualize>
   </sensor>
   ```

2. Verify topic publishing:
   ```bash
   ros2 topic echo /sensor_topic --field data 1  # Test single message
   ros2 topic hz /sensor_topic  # Check update rate
   ```

3. Check sensor mounting:
   ```xml
   <sensor name='camera'>
     <pose>0 0 0 0 0 0</pose>  <!-- Ensure proper mounting pose -->
   </sensor>
   ```

### Actuator Control Issues

#### Problem: Actuators not responding
**Symptoms**: Robot joints don't move despite commands

**Solutions**:
1. Check joint limits:
   ```xml
   <joint name='joint1' type='revolute'>
     <limit>
       <lower>-1.57</lower>
       <upper>1.57</upper>
       <effort>10</effort>
       <velocity>1</velocity>
     </limit>
   </joint>
   ```

2. Verify controller configuration:
   ```yaml
   # controller_manager.yaml
   controller_manager:
     ros__parameters:
       update_rate: 100  # Hz

   joint_state_broadcaster:
     type: joint_state_broadcaster/JointStateBroadcaster

   position_controller:
     type: position_controllers/JointGroupPositionController
     joints:
       - joint1
       - joint2
   ```

## Debugging Techniques

### ROS 2 Debugging

#### Enable detailed logging:
```bash
# Set logging level
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG

# Or set for specific node
ros2 run my_package my_node --ros-args --log-level DEBUG
```

#### Monitor system resources:
```bash
# Check topic rates
ros2 topic hz /my_topic

# Monitor node CPU usage
htop -p $(pgrep -f my_node)

# Check network usage
iftop -i lo  # For localhost
```

### Simulation Debugging

#### Gazebo debugging:
```bash
# Run with verbose output
gzserver --verbose

# Check physics statistics
gz stats

# Enable contact visualization
# In Gazebo GUI: View -> Contacts
```

#### Unity debugging:
```csharp
// Add debug visualization
void OnDrawGizmos()
{
    // Visualize transforms, rays, etc.
    Gizmos.color = Color.red;
    Gizmos.DrawRay(transform.position, transform.forward * 1f);
}
```

## Error Messages and Solutions

### Common Error Messages

#### "Failed to create subscriber"
**Cause**: Node not properly initialized
**Solution**: Ensure node is created before creating subscribers

#### "Joint limits exceeded"
**Cause**: Commanded position outside joint limits
**Solution**: Check joint limits and implement position clamping

#### "Transform not found"
**Cause**: TF tree not properly configured
**Solution**: Verify frame names and TF publishing

#### "Connection refused"
**Cause**: Service/server not available
**Solution**: Check if service is running and network configuration

### When to Restart

Sometimes a full restart is needed:
1. After major configuration changes
2. When nodes become unresponsive
3. After system crashes or freezes
4. When experiencing persistent connection issues

**Restart sequence**:
```bash
# Kill all ROS 2 processes
pkill -f ros

# For Gazebo
pkill -f gzserver
pkill -f gzclient

# Source everything fresh
source /opt/ros/humble/setup.bash
cd ~/ros2_ws && source install/setup.bash
```

## Getting Help

### Useful Commands for Diagnostics

```bash
# System information
ros2 doctor  # ROS 2 system check
ros2 run ros2run check  # Check ROS 2 installation

# Network diagnostics
ros2 topic list --verbose  # Detailed topic info
ros2 node info /node_name  # Node details

# Performance monitoring
ros2 run top ros_processes  # ROS-specific top
ros2 run plotjuggler plotjuggler  # Real-time plotting
```

### Community Resources

- **ROS Answers**: https://answers.ros.org/
- **Gazebo Answers**: https://answers.gazebosim.org/
- **Unity Forums**: https://forum.unity.com/
- **Isaac Sim Documentation**: NVIDIA developer documentation
- **GitHub Issues**: Check project repositories for known issues

This troubleshooting guide covers the most common issues encountered in robotics simulation and development. When facing a new problem, start with the basics: check connections, verify configurations, and ensure all required services are running.