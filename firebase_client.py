import logging
from pathlib import Path
from typing import Optional
import firebase_admin
from firebase_admin import credentials, storage

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    pass


class StorageConnectionError(Exception):
    pass


class FirebaseStorageClient:
    def __init__(self, bucket_name: str, credentials_path: str, cache_dir: str = "/app/model_cache"):
        self.bucket_name = bucket_name
        self.cache_dir = Path(cache_dir)
        self.bucket = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model cache directory: {self.cache_dir}")

        try:
            logger.info("Initializing Firebase Admin SDK...")
            if not firebase_admin._apps:
                cred = credentials.Certificate(credentials_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name
                })

            self.bucket = storage.bucket(bucket_name)
            logger.info(f"Connected to Firebase Storage bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise StorageConnectionError(f"Firebase initialization failed: {str(e)}")

    def get_cached_model_path(self, model_id: str) -> Optional[Path]:
        cached_path = self.cache_dir / model_id
        if cached_path.exists():
            logger.info(f"Model found in cache: {model_id}")
            return cached_path
        return None

    def download_model(self, model_id: str) -> Path:
        cached_path = self.get_cached_model_path(model_id)
        if cached_path:
            return cached_path

        try:
            firebase_path = f"policies/{model_id}"
            local_path = self.cache_dir / model_id

            logger.info(f"Downloading model from Firebase: {firebase_path}")
            blob = self.bucket.blob(firebase_path)

            if not blob.exists():
                logger.error(f"Model not found in Firebase Storage: {firebase_path}")
                raise ModelNotFoundError(f"Model '{model_id}' not found in storage")

            blob.download_to_filename(str(local_path))

            if not local_path.exists():
                raise StorageConnectionError(f"Downloaded file not found: {local_path}")

            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"Download complete: {model_id} ({file_size_mb:.2f} MB)")

            return local_path

        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise StorageConnectionError(f"Model download failed: {str(e)}")

    def list_available_models(self) -> list[str]:
        try:
            blobs = self.bucket.list_blobs(prefix="policies/")
            models = [blob.name.replace("policies/", "") for blob in blobs if blob.name.endswith(".pt")]
            logger.info(f"Found {len(models)} models in Firebase Storage")
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_cache_size_gb(self) -> float:
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt") if f.is_file())
        return total_size / (1024 ** 3)

    def clear_cache(self):
        for model_file in self.cache_dir.glob("*.pt"):
            try:
                model_file.unlink()
                logger.info(f"Removed cached model: {model_file.name}")
            except Exception as e:
                logger.error(f"Failed to remove {model_file.name}: {e}")
