#!/usr/bin/env python3
"""
Database migration script to add hashed_password column to users table
"""
import sys
import os

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import engine
from app.core.config import settings
from sqlalchemy import text
import bcrypt
import uuid


def add_hashed_password_column():
    """Add hashed_password column to users table"""
    
    with engine.begin() as conn:
        try:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = 'users' 
                AND column_name = 'hashed_password'
            """)).scalar()
            
            if result == 0:
                print("Adding hashed_password column to users table...")
                conn.execute(text("""
                    ALTER TABLE users 
                    ADD COLUMN hashed_password VARCHAR(255)
                """))
                print("✅ Column added successfully!")
            else:
                print("✅ Column hashed_password already exists!")
                
            # Update existing test users with hashed passwords
            print("Updating test users with hashed passwords...")
            
            # Hash passwords for test users
            admin_hash = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            user_hash = bcrypt.hashpw("user123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Check if admin user exists and update
            admin_exists = conn.execute(text("""
                SELECT COUNT(*) FROM users WHERE email = 'admin@example.com'
            """)).scalar()
            
            if admin_exists:
                conn.execute(text("""
                    UPDATE users 
                    SET hashed_password = :hash 
                    WHERE email = 'admin@example.com'
                """), {"hash": admin_hash})
                print("✅ Updated admin user password hash")
            else:
                # Create admin user
                conn.execute(text("""
                    INSERT INTO users (user_id, email, name, hashed_password, is_active, preferences, created_at)
                    VALUES (:user_id, :email, :name, :hash, :active, :prefs, NOW())
                """), {
                    "user_id": str(uuid.uuid4()),
                    "email": "admin@example.com",
                    "name": "Admin User",
                    "hash": admin_hash,
                    "active": True,
                    "prefs": "{}"
                })
                print("✅ Created admin user")
            
            # Check if test user exists and update
            user_exists = conn.execute(text("""
                SELECT COUNT(*) FROM users WHERE email = 'user@example.com'
            """)).scalar()
            
            if user_exists:
                conn.execute(text("""
                    UPDATE users 
                    SET hashed_password = :hash 
                    WHERE email = 'user@example.com'
                """), {"hash": user_hash})
                print("✅ Updated test user password hash")
            else:
                # Create test user
                conn.execute(text("""
                    INSERT INTO users (user_id, email, name, hashed_password, is_active, preferences, created_at)
                    VALUES (:user_id, :email, :name, :hash, :active, :prefs, NOW())
                """), {
                    "user_id": str(uuid.uuid4()),
                    "email": "user@example.com",
                    "name": "Test User",
                    "hash": user_hash,
                    "active": True,
                    "prefs": "{}"
                })
                print("✅ Created test user")
                
        except Exception as e:
            print(f"❌ Error during migration: {e}")
            raise


if __name__ == "__main__":
    print("Running database migration...")
    add_hashed_password_column()
    print("Migration completed!")
