import os

# Application settings are loaded during test collection. This bootstrap URL
# cannot create a file, and endpoint fixtures still override every DB session
# with a fresh temporary SQLite database.
os.environ.setdefault("DATABASE_URL", "sqlite://")
