from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, List
from uuid import UUID
from sqlalchemy import select, func
import logging

from schemas.chat import ChatRequest, ChatResponse, ChatSessionCreate, ChatSessionUpdate, ChatSessionResponse
from models.chat_session import ChatSession
from models.query import Query
from models.user import User
from db.session import get_db
from utils.dependencies import get_current_user_optional
from core.config import settings

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    session_data: ChatSessionCreate,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """
    Create a new chat session for the RAG chatbot.
    If user is authenticated, associate session with user, otherwise create anonymous session.
    """
    try:
        # Create new chat session
        user_id = current_user.id if current_user else None
        new_session = ChatSession(
            user_id=user_id,
            session_title=session_data.session_title
        )

        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)

        return ChatSessionResponse.from_orm(new_session)
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat session"
        )


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    chat_request: ChatRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """
    Process a chat query using RAG (Retrieval-Augmented Generation).
    This endpoint handles both regular queries and selected-text mode queries.
    """
    try:
        # Get or create chat session
        session_id = chat_request.session_id
        if session_id:
            # Retrieve existing session
            chat_session = await db.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
        else:
            # Create new session
            session_title = chat_request.message[:50] + "..." if len(chat_request.message) > 50 else chat_request.message
            user_id = current_user.id if current_user else None
            chat_session = ChatSession(
                user_id=user_id,
                session_title=session_title
            )
            db.add(chat_session)
            await db.commit()
            await db.refresh(chat_session)

        # Prepare query based on mode
        query_text = chat_request.message
        if chat_request.selected_text:
            # Selected-text mode: combine selected text with query
            query_text = f"Based on this text: {chat_request.selected_text}\n\nQuestion: {chat_request.message}"

        # For now, implement a simple response mechanism
        # In a full implementation, this would use semantic search and LLM
        if "week" in query_text.lower() or "module" in query_text.lower():
            # Simple response based on course structure
            llm_response = f"I found information about {query_text} in the Physical AI & Humanoid Robotics course materials. This query seems to relate to the course content. In a full implementation, I would search the indexed course materials and provide specific answers based on the book content."
        else:
            llm_response = f"I received your query: '{query_text}'. In a full implementation of the RAG system, I would search through the Physical AI & Humanoid Robotics course materials and provide you with accurate, context-aware responses based on the book content."

        # Create query record for logging
        new_query = Query(
            session_id=chat_session.id,
            user_id=current_user.id if current_user else None,
            query_text=chat_request.message,
            response_text=llm_response,
            selected_text=chat_request.selected_text
        )

        db.add(new_query)
        await db.commit()

        # Update session last activity
        # Note: We need to import func from sqlalchemy at the top of the file
        from sqlalchemy import func
        chat_session.last_activity_at = func.now()
        await db.commit()

        # For now, return a simple response with empty sources
        # In a full implementation, sources would come from the RAG search
        return ChatResponse(
            response=llm_response,
            session_id=chat_session.id,
            sources=[]
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat query"
        )


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """
    Get details of a specific chat session.
    Authenticated users can only access their own sessions, anonymous sessions can be accessed with session_id.
    """
    chat_session = await db.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Check if user has permission to access this session
    if chat_session.user_id and current_user:
        if chat_session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )
    elif chat_session.user_id and not current_user:
        # Authenticated session but no user logged in
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authentication required to access this session"
        )

    return ChatSessionResponse.from_orm(chat_session)


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(
    current_user: User = Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """
    List chat sessions for the current user.
    If user is not authenticated, returns empty list.
    """
    if not current_user:
        return []

    sessions = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.last_activity_at.desc())
    )

    return [ChatSessionResponse.from_orm(session) for session in sessions.scalars().all()]


@router.put("/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_chat_session(
    session_id: UUID,
    session_update: ChatSessionUpdate,
    current_user: User = Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """
    Update a chat session (e.g., rename title).
    """
    chat_session = await db.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Check if user has permission to update this session
    if chat_session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this session"
        )

    # Update session
    if session_update.session_title:
        chat_session.session_title = session_update.session_title

    await db.commit()
    await db.refresh(chat_session)

    return ChatSessionResponse.from_orm(chat_session)


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """
    Delete a chat session and all associated queries.
    """
    chat_session = await db.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Check if user has permission to delete this session
    if chat_session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this session"
        )

    # Delete session (and associated queries due to cascade)
    await db.delete(chat_session)
    await db.commit()

    return {"message": "Chat session deleted successfully"}