# -----------------------------
# PostgreSQL connection & session setup
# -----------------------------

import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from config import DB_URL
from db.models import Base

logger = logging.getLogger(__name__)

# Create engine (connect to DB)
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,  # Automatically retry failed connections
    pool_size=10,  # Number of connections to keep open
    max_overflow=20,  # Max additional connections
    pool_timeout=30,  # Timeout for getting a connection
    echo=False  # Set to True for SQL debugging
)

# Create tables if they don't exist
with engine.connect() as conn:
    Base.metadata.create_all(bind=conn)

# Create session factory with scoped sessions for thread safety
Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
SessionLocal = scoped_session(Session)


def get_session() -> scoped_session:
    """
    Get a new database session.

    Returns:
        scoped_session: New database session
    """
    return SessionLocal()


def close_session(session: scoped_session) -> None:
    """
    Close a database session.

    Args:
        session: Session to close
    """
    session.close()


# Event listener for cleanup on session close
@event.listens_for(SessionLocal, "close")
def remove_session(session):
    session.remove()
