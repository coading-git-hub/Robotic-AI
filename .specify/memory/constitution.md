# Physical AI & Humanoid Robotics Constitution

## Core Principles

### I. Technical Accuracy and Documentation Excellence
All content must be based on official documentation from authoritative sources (ROS 2, Gazebo, Unity, Isaac, Nav2, OpenAI, Qdrant, Neon). Every tutorial, code example, and explanation must be verified against official documentation and proven to work in practice.

### II. Educational Clarity and Accessibility
Content must be designed for students and newcomers to robotics. Complex concepts should be broken down into digestible lessons with clear explanations, step-by-step tutorials, and practical examples. All materials should be beginner-friendly while maintaining technical depth.

### III. Reproducibility and Consistency
Every tutorial, example, and procedure must be fully reproducible in a clean environment. The book and RAG system must maintain consistent content. All code examples should work when copy-pasted and all setups should be achievable from scratch.

### IV. Modularity and Structured Learning
The curriculum follows a modular approach with 4 core modules: ROS 2 fundamentals, Simulation environments (Gazebo & Unity), NVIDIA Isaac ecosystem, and Vision-Language-Action systems. Each module builds upon the previous ones while remaining self-contained.

### V. Integration and Practical Application
Focus on connecting different systems and technologies. The capstone project (Autonomous Humanoid) demonstrates integration of voice input, planning, navigation, object detection, and manipulation in a cohesive system.

### VI. Open Source and Community Standards
All code, documentation, and tools follow open-source best practices. Clear licensing, contribution guidelines, and community standards are maintained. All dependencies and tools are chosen with long-term sustainability in mind.

## Additional Constraints and Standards

### Technology Stack Requirements
- Docusaurus with Mintlify-style theme for documentation
- GitHub Pages for frontend hosting
- FastAPI backend for RAG system
- Qdrant Cloud for embeddings storage
- Neon Postgres for query logging
- ROS 2 Humble Hawksbill or newer
- NVIDIA Isaac ROS packages for perception
- Unity 2022.3 LTS or newer for simulation
- Python 3.10+ for all backend services

### Content Standards
- 120-200 pages equivalent in Docusaurus format
- Quarter Overview with 4 modules structure
- Each module contains Lessons → Headings → Sub-headings → Code + Examples
- Full Capstone project: Autonomous Humanoid with voice → plan → navigate → detect → manipulate pipeline
- Selected-text mode for RAG chatbot to ensure precise answers

### Deployment and Infrastructure
- Frontend deployed to GitHub Pages
- RAG backend deployed on Render/Fly.io
- All examples must work in clean ROS 2 + Isaac environment
- Continuous integration and deployment workflows established

## Development Workflow

### Feature Development Process
1. Create feature specification in `specs/<feature>/spec.md`
2. Develop architectural plan in `specs/<feature>/plan.md`
3. Break down into testable tasks in `specs/<feature>/tasks.md`
4. Implement with test-driven approach
5. Validate against acceptance criteria
6. Document in Docusaurus format
7. Add to RAG system for chatbot access

### Quality Gates
- All tutorials tested in fresh environments
- Code examples validated for correctness
- Cross-references between book and RAG system verified
- Performance benchmarks met for RAG system
- Security reviews for all deployed services

### Review Process
- Technical accuracy verified by domain experts
- Educational effectiveness validated by teaching staff
- Reproducibility confirmed on different machines
- Integration points tested across all systems

## Governance

The Physical AI & Humanoid Robotics project operates under these constitutional rules:
- All content must align with official documentation from authoritative sources
- Student accessibility is prioritized over technical completeness
- All examples and tutorials must be reproducible in clean environments
- Integration between book and RAG system is maintained consistently
- Capstone project serves as the ultimate integration test for all components
- All changes require validation against the core principles outlined above

**Version**: 1.0.0 | **Ratified**: 2025-12-12 | **Last Amended**: 2025-12-12
