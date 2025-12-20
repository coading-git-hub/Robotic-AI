# Data Model: Authentication and Personalized Content System

## User Profile Entity

**Description**: Core entity representing a registered user with authentication credentials and background information.

**Fields**:
- `id` (string/UUID): Unique identifier for the user
- `email` (string): User's email address (required, unique)
- `name` (string): User's display name (optional)
- `created_at` (timestamp): Account creation timestamp
- `updated_at` (timestamp): Last update timestamp
- `software_background` (enum): User's software experience level (required)
  - Values: "Beginner", "Frontend", "Backend", "AI"
- `hardware_background` (enum): User's hardware experience level (required)
  - Values: "Low-end PC", "Mid-range", "High-end", "GPU"
- `background_updated_at` (timestamp): Last time background info was updated

**Relationships**:
- One-to-many with User Activity logs (not implemented in this feature)
- One-to-many with Personalization History (not implemented in this feature)

**Validation Rules**:
- Email must be valid format and unique
- Software background must be one of the allowed enum values
- Hardware background must be one of the allowed enum values
- Both background fields are required during signup

## Personalization Session Entity

**Description**: Represents a personalization session when a user clicks the "Personalize" button for a specific chapter.

**Fields**:
- `id` (string/UUID): Unique identifier for the session
- `user_id` (string/UUID): Reference to the user who initiated personalization
- `chapter_id` (string): Identifier for the chapter being personalized
- `background_applied` (json): The user's background information used for this personalization
- `personalization_applied` (boolean): Whether personalization was successfully applied
- `created_at` (timestamp): When the personalization was initiated

**Relationships**:
- Belongs to User Profile (user_id foreign key)

**Validation Rules**:
- User must be authenticated to create a session
- Chapter ID must be valid
- Background information must exist for the user

## Database Schema (Neon Postgres)

```sql
-- Extending the Better Auth user table with background information
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS software_background VARCHAR(20);
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS hardware_background VARCHAR(20);
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS background_updated_at TIMESTAMP;

-- Add constraints for background fields
ALTER TABLE "user" ADD CONSTRAINT chk_software_background
  CHECK (software_background IN ('Beginner', 'Frontend', 'Backend', 'AI'));

ALTER TABLE "user" ADD CONSTRAINT chk_hardware_background
  CHECK (hardware_background IN ('Low-end PC', 'Mid-range', 'High-end', 'GPU'));

-- Create personalization session table
CREATE TABLE IF NOT EXISTS personalization_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
  chapter_id VARCHAR(100) NOT NULL,
  background_applied JSONB NOT NULL,
  personalization_applied BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_personalization_sessions_user_id ON personalization_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_personalization_sessions_chapter_id ON personalization_sessions(chapter_id);
CREATE INDEX IF NOT EXISTS idx_personalization_sessions_created_at ON personalization_sessions(created_at);
```

## API Data Transfer Objects

### User Registration Request
```json
{
  "email": "user@example.com",
  "password": "securePassword123",
  "name": "User Name",
  "software_background": "Beginner",
  "hardware_background": "Mid-range"
}
```

### User Profile Response
```json
{
  "id": "user-uuid",
  "email": "user@example.com",
  "name": "User Name",
  "software_background": "Beginner",
  "hardware_background": "Mid-range",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-01T00:00:00Z",
  "background_updated_at": "2023-01-01T00:00:00Z"
}
```

### Personalization Request
```json
{
  "chapter_id": "chapter-1-introduction",
  "user_background": {
    "software_background": "Beginner",
    "hardware_background": "Mid-range"
  }
}
```

### Personalization Response
```json
{
  "session_id": "session-uuid",
  "chapter_id": "chapter-1-introduction",
  "personalization_applied": true,
  "content_variants": {
    "difficulty_level": "beginner",
    "examples": ["simplified", "practical"],
    "hardware_specific_content": false
  }
}
```