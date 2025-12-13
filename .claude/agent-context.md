# Claude Agent Context: Physical AI & Humanoid Robotics

## Project Overview
The Physical AI & Humanoid Robotics project is a comprehensive educational platform consisting of:
- A 13-week course covering Physical AI, ROS 2, simulation, NVIDIA Isaac, and Vision-Language-Action systems
- Integrated RAG chatbot functionality for interactive learning
- Docusaurus-based documentation with embedded interactive elements
- Capstone project: Autonomous humanoid executing voice → plan → navigate → detect → manipulate tasks

## Technical Stack
- **Frontend**: Docusaurus with Mintlify-style theme
- **Backend**: FastAPI for RAG system
- **Vector Database**: Qdrant Cloud for embeddings storage
- **Logging**: Neon Postgres for query logging
- **Robotics**: ROS 2 Humble Hawksbill, Gazebo, Unity, NVIDIA Isaac
- **AI Components**: OpenAI Whisper, LLMs for cognitive planning, VLA systems
- **Deployment**: GitHub Pages (frontend), Render/Fly.io (backend)

## Course Structure (13 Weeks)
1. **Weeks 1-2**: Foundations of Physical AI and Embodied Intelligence
2. **Weeks 3-5**: ROS 2 Fundamentals (nodes, topics, services, actions)
3. **Weeks 6-8**: Simulation and Digital Twins (Gazebo, Unity, Isaac Sim)
4. **Weeks 9-11**: NVIDIA Isaac and Advanced Perception
5. **Weeks 12-13**: Vision-Language-Action (VLA) and Capstone Project

## Key Technologies
- **Simulation**: Gazebo (physics), Unity (rendering), Isaac Sim (perception)
- **AI/ML**: OpenAI Whisper, LLMs via API, Isaac perception pipelines
- **RAG System**: FastAPI + Qdrant + Neon Postgres
- **Documentation**: Docusaurus with embedded chatbot widget

## RAG Chatbot Features
- Embedded in Docusaurus pages with floating widget interface
- Selected-text mode: responds based only on user-selected content
- Semantic search across book content
- Session management for conversation continuity
- Feedback system for response quality improvement

## Content Standards
- 120-200 pages equivalent in Docusaurus format
- 40% theory, 60% hands-on implementation balance
- All examples fully reproducible in clean environments
- Integration checkpoints connecting modules
- Capstone project integrating all components

## Development Workflow
- Spec-Kit Plus methodology: spec → plan → tasks → implementation
- Module-by-module content generation with Claude Code
- Quality validation: technical correctness, runnable code, accurate workflows
- Docusaurus-ready Markdown with diagrams, tables, and interactive elements

## File Structure
```
book/
├── architecture/          # System architecture and implementation plans
├── content/              # Weekly modules and lessons
├── research/             # Research documents and decisions
├── design/               # Data models, contracts, quickstart guides
├── rag-system/           # RAG backend and frontend components
└── assets/               # Code examples, diagrams, interactive elements

specs/
├── course/               # Course-level specification
├── 001-ros2-nervous-system/    # Module 1 spec
├── 002-gazebo-unity-digital-twin/ # Module 2 spec
├── 003-isaac-ai-brain/         # Module 3 spec
└── 004-vla-humanoid/           # Module 4 spec
```

## Important Files
- `book/architecture/implementation-plan.md` - Main implementation plan
- `book/research/research.md` - Research and decision documentation
- `book/design/data-model.md` - Data model for all entities
- `book/contracts/rag-api-contracts.md` - API contracts for RAG system
- `book/design/quickstart.md` - Quickstart guide for users
- `specs/course/spec.md` - Course-level specification

## Constitutional Principles
- Technical accuracy from authoritative sources (ROS 2, Gazebo, Unity, Isaac, etc.)
- Educational clarity and accessibility for students
- Reproducibility and consistency across all examples
- Modularity and structured learning approach
- Integration and practical application focus
- Open source and community standards adherence

## Current Status
- Course specification complete with all ambiguities resolved
- Implementation plan created with research, design, and development phases
- Data model and API contracts defined
- RAG system architecture established
- Ready for content generation and system implementation phase