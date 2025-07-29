#!/usr/bin/env python3
"""
Development script to reset the database by dropping and recreating all tables.
WARNING: This will delete all data in your database.
"""
import sys
import os

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import drop_tables, create_tables
from app.core.logging import logger

def reset_database():
    """Drops all tables and recreates them based on the current models."""
    logger.warning("--- DATABASE RESET SCRIPT ---")
    logger.warning("WARNING: This will delete all data in your database.")
    
    confirm = input("Are you sure you want to continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Database reset cancelled.")
        return

    try:
        logger.info("Dropping all tables...")
        drop_tables()
        logger.info("✅ Tables dropped successfully.")
        
        logger.info("Creating all tables based on current models...")
        create_tables()
        logger.info("✅ Tables created successfully.")
        
        print("\nDatabase has been reset.")
        
    except Exception as e:
        logger.error(f"❌ An error occurred during database reset: {e}")
        print("\nDatabase reset failed. Please check the logs.")

if __name__ == "__main__":
    reset_database()