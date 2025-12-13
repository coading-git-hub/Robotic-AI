# Quickstart Guide: Physical AI & Humanoid Robotics Book with RAG Chatbot

**Feature**: course-physical-ai-humanoid
**Created**: 2025-12-12
**Status**: Complete
**Plan**: [implementation-plan.md](../architecture/implementation-plan.md)

## Overview

This quickstart guide provides the essential information needed to get started with the Physical AI & Humanoid Robotics book and its integrated RAG chatbot system. Follow these steps to begin your journey in humanoid robotics education.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- **Hardware**:
  - **Minimum**: 16GB RAM, 8-core CPU, 500GB SSD
  - **Recommended**: RTX 3080/4090+, 32GB+ RAM, 1TB+ SSD
- **Software**: Python 3.10+, Git, Docker (optional but recommended)

### Software Stack
- ROS 2 Humble Hawksbill
- Gazebo Garden or Fortress
- Unity 2022.3 LTS (for high-fidelity rendering)
- NVIDIA Isaac Sim (for advanced perception)
- FastAPI for backend services
- Node.js/npm for frontend development

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/physical-ai/humanoid-robotics.git
cd humanoid-robotics
```

### 2. Set up ROS 2 Environment
```bash
# Install ROS 2 Humble (Ubuntu)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-colcon-common-extensions
source /opt/ros/humble/setup.bash
```

### 3. Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### 4. Set up RAG System Backend
```bash
# Navigate to RAG backend directory
cd rag-system/backend

# Install FastAPI dependencies
pip install "fastapi[all]" uvicorn qdrant-client python-dotenv psycopg2-binary openai

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### 5. Start the Development Environment
```bash
# Terminal 1: Start ROS 2
source /opt/ros/humble/setup.bash
source install/setup.bash

# Terminal 2: Start RAG backend
cd rag-system/backend
uvicorn main:app --reload --port 8000

# Terminal 3: Start Docusaurus frontend
cd book-frontend
npm install
npm start
```

## Book Structure

### Module Organization (13-Week Course)
```
book/
├── week-01-02-foundations/     # Physical AI & Embodied Intelligence
├── week-03-05-ros2/           # ROS 2 Fundamentals
├── week-06-08-simulation/     # Gazebo, Unity, Isaac Sim
├── week-09-11-isaac/          # NVIDIA Isaac & Advanced Perception
└── week-12-13-vla-capstone/   # Vision-Language-Action & Capstone
```

### Content Format
Each module follows this structure:
- **Lessons** → **Headings** → **Sub-headings** → **Code + Examples**
- Theory sections with practical exercises
- Integration checkpoints connecting modules
- Assessment components for each week

## Using the RAG Chatbot

### 1. Access the Chat Interface
- The chatbot is embedded in every page of the Docusaurus documentation
- Click the floating widget in the bottom-right corner
- Select text on any page to ask context-specific questions

### 2. Query Examples
```
"Explain the difference between ROS 2 nodes and services"
"Show me the code example for creating a publisher node"
"How does Isaac Sim integrate with ROS 2?"
"What are the key components of the VLA pipeline?"
```

### 3. Selected-Text Mode
- Highlight text on any page
- Click "Ask about this" to get responses based only on the selected content
- Ensures precise, context-aware answers

## First Steps

### Week 1: Getting Started
1. **Read**: Module 1, Lesson 1 - "Introduction to Physical AI"
2. **Practice**: Set up your development environment
3. **Explore**: Navigate the book structure and test the RAG chatbot
4. **Exercise**: Complete the basic ROS 2 node creation example

### Week 2: Foundations
1. **Read**: Module 1, Lesson 2 - "Embodied Intelligence Concepts"
2. **Practice**: Run basic Gazebo simulation with humanoid model
3. **Exercise**: Implement simple sensor data processing
4. **Assess**: Complete Week 1-2 quiz

## Key Commands

### ROS 2 Development
```bash
# Build the workspace
colcon build

# Source the workspace
source install/setup.bash

# Run a ROS 2 node
ros2 run package_name node_name

# Launch a system
ros2 launch package_name launch_file.launch.py
```

### RAG System Management
```bash
# Start the RAG backend
cd rag-system/backend
uvicorn main:app --reload

# Index book content
python scripts/index_content.py

# Test the API
curl -X POST http://localhost:8000/chat/query -H "Content-Type: application/json" -d '{"query": "test"}'
```

### Docusaurus Frontend
```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Serve production build
npm run serve
```

## Troubleshooting

### Common Issues

**Issue**: ROS 2 nodes not communicating
**Solution**: Ensure all terminals source the workspace: `source install/setup.bash`

**Issue**: RAG chatbot not responding
**Solution**: Check if the backend is running: `curl http://localhost:8000/health`

**Issue**: Slow simulation performance
**Solution**: Reduce physics update rate or simplify environment complexity

**Issue**: Docusaurus build errors
**Solution**: Clear cache: `npm run clear` and reinstall dependencies

### Getting Help
- Use the RAG chatbot for content-specific questions
- Check the troubleshooting section in each module
- Review the API documentation at `/docs` endpoint
- Join the community Discord for real-time support

## Development Workflow

### Creating New Content
1. Create new lesson files in the appropriate week directory
2. Follow the existing Markdown format with proper frontmatter
3. Add code examples in the assets directory
4. Update the sidebar configuration
5. Test the RAG integration by indexing new content

### Adding New API Endpoints
1. Define the endpoint in the FastAPI application
2. Add proper request/response models
3. Implement business logic
4. Add to the API contract documentation
5. Test with the integrated frontend

## Next Steps

1. **Complete Week 1-2**: Establish your development environment and foundational knowledge
2. **Join the Community**: Access additional resources and support
3. **Set Up Hardware**: If available, configure Jetson Orin AGX for real-world deployment
4. **Plan Your Capstone**: Begin thinking about your final project ideas
5. **Track Progress**: Use the built-in progress tracking system

## Support Resources

- **Documentation**: Full API docs at `/docs` endpoint
- **Examples**: Code examples in `book/assets/code-examples/`
- **Community**: Discord server and GitHub Discussions
- **Video Tutorials**: Embedded videos in relevant sections
- **Office Hours**: Weekly live Q&A sessions (schedule in Week 1)

This quickstart guide provides the foundation for your Physical AI & Humanoid Robotics journey. The integrated RAG chatbot is available throughout your learning experience to provide immediate, context-aware assistance based on the book content.