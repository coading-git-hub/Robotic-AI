# Data Model: Physical AI & Humanoid Robotics Book with RAG Chatbot

**Feature**: course-physical-ai-humanoid
**Created**: 2025-12-12
**Status**: Complete
**Plan**: [implementation-plan.md](../architecture/implementation-plan.md)

## Overview

This data model defines the entities and relationships for the Physical AI & Humanoid Robotics book with integrated RAG chatbot system. The model supports content management, user progress tracking, and RAG system functionality while maintaining consistency with the course structure.

## Content Entities

### Module
- **id**: UUID
- **title**: String (e.g., "Foundations of Physical AI", "ROS 2 Fundamentals")
- **number**: Integer (1-13 for weeks)
- **description**: Text
- **learning_objectives**: Array of strings
- **duration_hours**: Integer (estimated completion time)
- **prerequisites**: Array of strings (other modules or concepts)
- **created_at**: DateTime
- **updated_at**: DateTime
- **status**: Enum ("draft", "review", "published", "archived")
- **relationships**: lessons (one-to-many)

### Lesson
- **id**: UUID
- **module_id**: UUID (foreign key to Module)
- **title**: String
- **position**: Integer (order within module)
- **description**: Text
- **content_type**: Enum ("theory", "hands_on", "assessment", "integration")
- **duration_minutes**: Integer
- **learning_outcomes**: Array of strings
- **resources**: Array of strings (file paths, URLs)
- **created_at**: DateTime
- **updated_at**: DateTime
- **status**: Enum ("draft", "review", "published", "archived")
- **relationships**: sections (one-to-many), exercises (one-to-many)

### Section
- **id**: UUID
- **lesson_id**: UUID (foreign key to Lesson)
- **title**: String
- **position**: Integer (order within lesson)
- **content**: Text (Markdown format)
- **content_type**: Enum ("text", "code", "diagram", "exercise", "example")
- **difficulty_level**: Enum ("beginner", "intermediate", "advanced")
- **estimated_reading_time**: Integer (minutes)
- **requires_simulation**: Boolean
- **requires_hardware**: Boolean
- **created_at**: DateTime
- **updated_at**: DateTime
- **relationships**: code_examples (one-to-many), diagrams (one-to-many)

### CodeExample
- **id**: UUID
- **section_id**: UUID (foreign key to Section)
- **title**: String
- **description**: Text
- **code**: Text (source code content)
- **language**: String (python, launch, etc.)
- **file_path**: String (relative path in assets)
- **runnable**: Boolean
- **expected_output**: Text
- **dependencies**: Array of strings (required packages, nodes)
- **created_at**: DateTime
- **updated_at**: DateTime

### Exercise
- **id**: UUID
- **lesson_id**: UUID (foreign key to Lesson)
- **title**: String
- **description**: Text
- **instructions**: Text (Markdown format)
- **difficulty_level**: Enum ("beginner", "intermediate", "advanced")
- **estimated_completion_time**: Integer (minutes)
- **solution**: Text (Markdown format)
- **assessment_criteria**: Array of strings
- **requires_simulation**: Boolean
- **requires_hardware**: Boolean
- **created_at**: DateTime
- **updated_at**: DateTime

## User Entities

### User
- **id**: UUID
- **username**: String (unique)
- **email**: String (unique, validated)
- **role**: Enum ("student", "instructor", "admin")
- **first_name**: String
- **last_name**: String
- **institution**: String (optional)
- **programming_level**: Enum ("beginner", "intermediate", "advanced")
- **robotics_experience**: Enum ("none", "basic", "intermediate", "advanced")
- **created_at**: DateTime
- **updated_at**: DateTime
- **last_login_at**: DateTime (nullable)
- **relationships**: progress_records (one-to-many), queries (one-to-many)

### ProgressRecord
- **id**: UUID
- **user_id**: UUID (foreign key to User)
- **module_id**: UUID (foreign key to Module)
- **lesson_id**: UUID (foreign key to Lesson) (nullable)
- **section_id**: UUID (foreign key to Section) (nullable)
- **progress_type**: Enum ("module", "lesson", "section")
- **status**: Enum ("not_started", "in_progress", "completed", "review_needed")
- **completion_percentage**: Decimal (0.00 to 1.00)
- **time_spent_minutes**: Integer
- **score**: Decimal (0.00 to 1.00) (nullable)
- **attempts_count**: Integer
- **last_accessed_at**: DateTime
- **completed_at**: DateTime (nullable)
- **created_at**: DateTime
- **updated_at**: DateTime

## RAG System Entities

### ContentChunk
- **id**: UUID
- **content_id**: UUID (references Module, Lesson, or Section ID)
- **content_type**: Enum ("module", "lesson", "section")
- **text_content**: Text (the actual content chunk)
- **embedding_vector**: Array of floats (for similarity search)
- **token_count**: Integer
- **semantic_boundary**: Boolean (indicates if this is a semantic boundary)
- **created_at**: DateTime
- **updated_at**: DateTime
- **relationships**: embeddings (one-to-many)

### Query
- **id**: UUID
- **user_id**: UUID (foreign key to User) (nullable for anonymous)
- **session_id**: String (for tracking conversation context)
- **input_text**: Text (user's query)
- **selected_text**: Text (text selected by user for context)
- **query_type**: Enum ("general", "selected_text", "exercise_help", "code_explanation")
- **response_text**: Text (RAG system response)
- **retrieved_chunks**: Array of UUIDs (references to ContentChunk)
- **confidence_score**: Decimal (0.00 to 1.00)
- **response_time_ms**: Integer
- **feedback_rating**: Integer (1-5 scale) (nullable)
- **feedback_comment**: Text (nullable)
- **created_at**: DateTime
- **updated_at**: DateTime

### ChatSession
- **id**: UUID
- **user_id**: UUID (foreign key to User) (nullable for anonymous)
- **session_title**: String (auto-generated from first query)
- **active**: Boolean
- **created_at**: DateTime
- **updated_at**: DateTime
- **last_activity_at**: DateTime
- **relationships**: queries (one-to-many)

## System Entities

### SystemConfig
- **id**: UUID
- **config_key**: String (unique)
- **config_value**: Text
- **data_type**: Enum ("string", "integer", "boolean", "json")
- **description**: Text
- **category**: String (e.g., "rag", "content", "user", "system")
- **created_at**: DateTime
- **updated_at**: DateTime

### Notification
- **id**: UUID
- **user_id**: UUID (foreign key to User) (nullable for system notifications)
- **title**: String
- **message**: Text
- **notification_type**: Enum ("info", "warning", "error", "progress", "deadline")
- **priority**: Enum ("low", "medium", "high", "critical")
- **read_status**: Boolean
- **target_url**: String (nullable, URL to relevant content)
- **expires_at**: DateTime (nullable)
- **created_at**: DateTime
- **updated_at**: DateTime

## Integration Entities

### ROS2Component
- **id**: UUID
- **name**: String (e.g., "navigation_node", "perception_pipeline")
- **component_type**: Enum ("node", "package", "launch_file", "action", "service")
- **description**: Text
- **dependencies**: Array of strings
- **parameters**: JSON object
- **topics_subscribed**: Array of strings
- **topics_published**: Array of strings
- **services_used**: Array of strings
- **actions_used**: Array of strings
- **created_at**: DateTime
- **updated_at**: DateTime
- **relationships**: code_examples (many-to-many)

### SimulationEnvironment
- **id**: UUID
- **name**: String (e.g., "basic_gazebo_world", "unity_indoor_scene")
- **platform**: Enum ("gazebo", "unity", "isaac_sim")
- **description**: Text
- **world_file_path**: String
- **robot_model**: String
- **sensors_configured**: Array of strings
- **initial_conditions**: JSON object
- **performance_metrics**: JSON object
- **created_at**: DateTime
- **updated_at**: DateTime
- **relationships**: lessons (many-to-many)

## State Transitions

### ProgressRecord State Transitions
- `not_started` → `in_progress` (when user starts interacting with content)
- `in_progress` → `completed` (when user completes the content)
- `in_progress` → `review_needed` (when assessment score is below threshold)
- `completed` → `in_progress` (when user revisits content)
- `review_needed` → `completed` (when user successfully revisits content)

### Query State Management
- New queries are created with session context
- Confidence scores are calculated based on embedding similarity
- Responses are generated using selected text context when applicable
- Feedback is collected to improve future responses

## Validation Rules

### Content Validation
- Module numbers must be unique within the course (1-13)
- Lesson positions must be unique within each module
- Section positions must be unique within each lesson
- Content must not exceed maximum size limits for RAG processing
- Code examples must be validated for syntax and dependencies

### User Progress Validation
- Progress records must follow the logical sequence (module → lesson → section)
- Completion percentages must be between 0 and 100
- Score values must be between 0 and 1
- Time spent must be realistic for content type

### RAG System Validation
- Content chunks must not exceed maximum token limits
- Embedding vectors must have consistent dimensions
- Query responses must be based only on selected content when in selected-text mode
- Session management must maintain conversation context appropriately

## Relationships Summary

### Content Hierarchy
- Module (1) → (Many) Lesson → (Many) Section → (Many) CodeExample
- Module (1) → (Many) Lesson → (Many) Exercise

### User Interactions
- User (1) → (Many) ProgressRecord
- User (1) → (Many) Query → (Many) ContentChunk (via retrieval)

### System Integration
- Section (1) → (Many) CodeExample, Diagram
- Lesson (1) → (Many) Exercise
- User (1) → (Many) ChatSession → (Many) Query
- ContentChunk (1) → (Many) Embedding (for search)

This data model supports the complete functionality of the Physical AI & Humanoid Robotics book with integrated RAG chatbot, ensuring proper content organization, user progress tracking, and intelligent query handling while maintaining consistency with the educational objectives and technical requirements.