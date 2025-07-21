import os
import asyncio
import aiofiles
from typing import Optional, BinaryIO
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime

from ..core.config import settings
from ..core.logging import logger


class StorageInterface(ABC):
    """Abstract interface for file storage"""
    
    @abstractmethod
    async def store_file(self, file_content: bytes, file_path: str) -> str:
        """Store file and return storage path"""
        pass
    
    @abstractmethod
    async def get_file(self, file_path: str) -> bytes:
        """Retrieve file content"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file"""
        pass
    
    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists"""
        pass


class LocalStorageService(StorageInterface):
    """Local file system storage implementation"""
    
    def __init__(self, base_path: str = "storage/documents"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    async def store_file(self, file_content: bytes, file_path: str) -> str:
        """Store file locally"""
        try:
            full_path = os.path.join(self.base_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"Stored file locally: {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"Error storing file locally: {e}")
            raise
    
    async def get_file(self, file_path: str) -> bytes:
        """Retrieve file from local storage"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            return content
        except Exception as e:
            logger.error(f"Error retrieving file: {e}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from local storage"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists locally"""
        return os.path.exists(file_path)


class GCSStorageService(StorageInterface):
    """Google Cloud Storage implementation"""
    
    def __init__(self):
        self.bucket_name = settings.gcs_bucket_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize GCS client"""
        try:
            from google.cloud import storage
            
            if settings.gcs_credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.gcs_credentials_path
            
            self.client = storage.Client()
            logger.info("GCS client initialized successfully")
            
        except ImportError:
            logger.error("Google Cloud Storage library not available")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
    
    async def store_file(self, file_content: bytes, file_path: str) -> str:
        """Store file in GCS"""
        if not self.client or not self.bucket_name:
            raise Exception("GCS not properly configured")
        
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(file_path)
            
            # Use asyncio to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, blob.upload_from_string, file_content
            )
            
            storage_path = f"gs://{self.bucket_name}/{file_path}"
            logger.info(f"Stored file in GCS: {storage_path}")
            return storage_path
            
        except Exception as e:
            logger.error(f"Error storing file in GCS: {e}")
            raise
    
    async def get_file(self, file_path: str) -> bytes:
        """Retrieve file from GCS"""
        if not self.client:
            raise Exception("GCS not properly configured")
        
        try:
            # Extract blob path from full GCS path
            if file_path.startswith("gs://"):
                blob_path = file_path.replace(f"gs://{self.bucket_name}/", "")
            else:
                blob_path = file_path
            
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            content = await asyncio.get_event_loop().run_in_executor(
                None, blob.download_as_bytes
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error retrieving file from GCS: {e}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from GCS"""
        if not self.client:
            return False
        
        try:
            # Extract blob path from full GCS path
            if file_path.startswith("gs://"):
                blob_path = file_path.replace(f"gs://{self.bucket_name}/", "")
            else:
                blob_path = file_path
            
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            await asyncio.get_event_loop().run_in_executor(None, blob.delete)
            
            logger.info(f"Deleted file from GCS: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from GCS: {e}")
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in GCS"""
        if not self.client:
            return False
        
        try:
            # Extract blob path from full GCS path
            if file_path.startswith("gs://"):
                blob_path = file_path.replace(f"gs://{self.bucket_name}/", "")
            else:
                blob_path = file_path
            
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            exists = await asyncio.get_event_loop().run_in_executor(
                None, blob.exists
            )
            
            return exists
            
        except Exception as e:
            logger.error(f"Error checking file existence in GCS: {e}")
            return False


class StorageService:
    """Main storage service that delegates to the appropriate implementation"""
    
    def __init__(self):
        self.storage = self._get_storage_implementation()
    
    def _get_storage_implementation(self) -> StorageInterface:
        """Get the appropriate storage implementation based on configuration"""
        if settings.storage_type == "gcs":
            return GCSStorageService()
        else:
            return LocalStorageService()
    
    async def store_file(self, file_content: bytes, file_path: str) -> str:
        """Store file using configured storage backend"""
        # Add timestamp and hash to avoid collisions
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(file_content).hexdigest()[:8]
        
        # Ensure unique file path
        path_parts = file_path.split('/')
        filename = path_parts[-1]
        directory = '/'.join(path_parts[:-1])
        
        unique_filename = f"{timestamp}_{content_hash}_{filename}"
        unique_path = f"{directory}/{unique_filename}" if directory else unique_filename
        
        return await self.storage.store_file(file_content, unique_path)
    
    async def get_file(self, file_path: str) -> bytes:
        """Retrieve file"""
        return await self.storage.get_file(file_path)
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file"""
        return await self.storage.delete_file(file_path)
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists"""
        return await self.storage.file_exists(file_path)
