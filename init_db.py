import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure app module resolves if run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db_models import Base
from app.routes.database import engine


async def init_models():
    print("Creating database tables...")
    async with engine.begin() as conn:
        # Create all tables (does not drop existing ones)
        await conn.run_sync(Base.metadata.create_all)

        # Backfill schema changes for existing databases
        await conn.exec_driver_sql("ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT;")
        await conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS photo_object_name TEXT;"
        )
        await conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS phone_number TEXT;"
        )
        await conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS years_of_experience SMALLINT;"
        )
        await conn.exec_driver_sql("ALTER TABLE users ADD COLUMN IF NOT EXISTS acres FLOAT;")
        await conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS primary_crops JSONB;"
        )
        await conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS soil_type TEXT;"
        )
    print("[OK] Tables created successfully!")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set. Please check your .env file.")
        sys.exit(1)

    asyncio.run(init_models())
