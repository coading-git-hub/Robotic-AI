// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Week 1-2: Foundations of Physical AI',
      items: [
        'week-1-2/physical-ai-intro',
        'week-1-2/embodied-intelligence',
        'week-1-2/sensor-systems',
      ],
    },
    {
      type: 'category',
      label: 'Week 3-5: ROS 2 Fundamentals',
      items: [
        'week-3-5/ros2-architecture',
        'week-3-5/nodes-topics-services',
        'week-3-5/ros2-packages',
      ],
    },
    {
      type: 'category',
      label: 'Week 6-8: Simulation and Digital Twins',
      items: [
        'week-6-8/gazebo-simulation',
        'week-6-8/unity-rendering',
        'week-6-8/isaac-sim',
      ],
    },
    {
      type: 'category',
      label: 'Week 9-11: NVIDIA Isaac and Advanced Perception',
      items: [
        'week-9-11/isaac-ros',
        'week-9-11/vslam-navigation',
        'week-9-11/sim-to-real',
      ],
    },
    {
      type: 'category',
      label: 'Week 12-13: Vision-Language-Action and Capstone',
      items: [
        'week-12-13/vla-integration',
        'week-12-13/capstone-project',
        'week-12-13/autonomous-humanoid',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/ros2-cheatsheet',
        'appendices/simulation-tips',
        'appendices/troubleshooting',
      ],
    },
  ],
};

export default sidebars;