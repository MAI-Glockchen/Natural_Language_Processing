# -----------------------------
# PostgreSQL connection & session setup
# -----------------------------

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DB_URL
from db.models import Base

# Create engine (connect to DB)
engine = create_engine(DB_URL)

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Create session factory
Session = sessionmaker(bind=engine)
session = Session()