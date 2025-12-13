import React, { useState, useEffect } from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <header className={`${styles.heroBanner} ${isVisible ? styles.fadeIn : ''}`}>
      <div className={styles.heroContent}>
        <h1 className={styles.heroTitle}>
          Welcome to the Future of Robotics
        </h1>
        <p className={styles.heroSubtitle}>
          Dive into the cutting-edge world where AI meets physical reality
        </p>
        <div className={styles.heroButtons}>
          <Link
            className={styles.primaryButton}
            to="/docs/intro"
          >
            Get Started
          </Link>
          <Link
            className={styles.secondaryButton}
            to="/blog"
          >
            Read Blog
          </Link>
        </div>
      </div>
      <div className={styles.heroBackground}>
        <div className={styles.gradientOrb1}></div>
        <div className={styles.gradientOrb2}></div>
        <div className={styles.gradientOrb3}></div>
      </div>
    </header>
  );
}

function CourseModule({ title, duration, description, link, icon, color }) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className={`${styles.moduleCard} ${isHovered ? styles.hovered : ''}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{ '--accent-color': color }}
    >
      <div className={styles.moduleIcon}>
        <span className={styles.iconEmoji}>{icon}</span>
      </div>
      <div className={styles.moduleContent}>
        <h3 className={styles.moduleTitle}>{title}</h3>
        <span className={styles.moduleDuration}>{duration}</span>
        <p className={styles.moduleDescription}>{description}</p>
        <Link to={link} className={styles.moduleLink}>
          Learn More →
        </Link>
      </div>
      <div className={styles.moduleGlow}></div>
    </div>
  );
}

function CourseModules() {
  const modules = [
    {
      title: 'Physical AI & Foundations',
      duration: 'Weeks 1-2',
      description: 'Understanding the principles of Physical AI and embodied intelligence that form the foundation of intelligent humanoid systems.',
      link: '/docs/week-1-2/physical-ai-intro',
      icon: '🧠',
      color: '#FF6B6B'
    },
    {
      title: 'ROS 2 Fundamentals',
      duration: 'Weeks 3-5',
      description: 'Master the Robot Operating System for communication, control, and coordination of complex robotic systems.',
      link: '/docs/week-3-5/ros2-architecture',
      icon: '🤖',
      color: '#4ECDC4'
    },
    {
      title: 'Simulation & Digital Twins',
      duration: 'Weeks 6-8',
      description: 'Create realistic simulation environments using Gazebo, Unity, and NVIDIA Isaac Sim for safe development.',
      link: '/docs/week-6-8/gazebo-simulation',
      icon: '🌍',
      color: '#45B7D1'
    },
    {
      title: 'Advanced Perception',
      duration: 'Weeks 9-11',
      description: 'Implement NVIDIA Isaac tools for vision-based navigation, SLAM, and sim-to-real transfer techniques.',
      link: '/docs/week-9-11/isaac-ros',
      icon: '👁️',
      color: '#96CEB4'
    },
    {
      title: 'Vision-Language-Action',
      duration: 'Weeks 12-13',
      description: 'Build intelligent systems that understand voice commands and execute complex autonomous behaviors.',
      link: '/docs/week-12-13/vla-integration',
      icon: '🗣️',
      color: '#FFEAA7'
    },
    {
      title: 'Capstone Project',
      duration: 'Final Project',
      description: 'Integrate all concepts into a complete autonomous humanoid robot capable of voice interaction.',
      link: '/docs/week-12-13/capstone-project',
      icon: '🏆',
      color: '#DDA0DD'
    }
  ];

  return (
    <section className={styles.modulesSection}>
      <div className={styles.container}>
        <h2 className={styles.sectionTitle}>🚀 Course Modules</h2>
        <div className={styles.modulesGrid}>
          {modules.map((module, index) => (
            <CourseModule key={index} {...module} />
          ))}
        </div>
      </div>
    </section>
  );
}

function InteractiveFeature() {
  return (
    <section className={styles.featureSection}>
      <div className={styles.container}>
        <div className={styles.featureContent}>
          <div className={styles.featureText}>
            <h2 className={styles.featureTitle}>🎯 Interactive Learning Experience</h2>
            <p className={styles.featureDescription}>
              This course features an integrated <strong>RAG (Retrieval-Augmented Generation)</strong> chatbot
              that provides immediate assistance based on the course content. Use the chatbot to clarify concepts,
              get code explanations, or explore related topics in more depth.
            </p>
            <div className={styles.featureStats}>
              <div className={styles.stat}>
                <span className={styles.statNumber}>13</span>
                <span className={styles.statLabel}>Weeks</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNumber}>6</span>
                <span className={styles.statLabel}>Modules</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNumber}>∞</span>
                <span className={styles.statLabel}>Possibilities</span>
              </div>
            </div>
          </div>
          <div className={styles.featureVisual}>
            <div className={styles.chatbotMockup}>
              <div className={styles.chatBubble}>
                <span>🤖</span>
                How can I help you with Physical AI today?
              </div>
              <div className={styles.chatInput}>
                <span>Type your question...</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Comprehensive Course on Physical AI, ROS 2, Simulation, and Autonomous Robotics"
    >
      <HomepageHeader />
      <main>
        <CourseModules />
        <InteractiveFeature />
      </main>
    </Layout>
  );
}