"""
File management system with security and cleanup features.
"""

import os
import time
import threading
import mimetypes
from pathlib import Path
from typing import Set, Optional
import uuid
import hashlib


class FileManager:
    """
    Manages file uploads, downloads, and cleanup with security features.
    """

    # Allowed file types and extensions
    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    ALLOWED_BIN_EXTENSIONS = {".bin"}

    # Maximum file size (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024

    # File cleanup timeout (3 minutes)
    CLEANUP_TIMEOUT = 3 * 60

    def __init__(self, upload_dir: Path, output_dir: Path):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track files for cleanup
        self.file_tracker: dict = {}
        self._reload_existing_files()
        self._start_cleanup_thread()

    def _reload_existing_files(self):
        for dir_path in [self.upload_dir, self.output_dir]:
            for file in dir_path.glob("*"):
                file_age = time.time() - file.stat().st_mtime
                if file_age > self.CLEANUP_TIMEOUT:
                    self._safe_remove_file(file)
                else:
                    self.file_tracker[str(file)] = time.time() - file_age

    def _start_cleanup_thread(self):
        """Start background thread for file cleanup."""

        def cleanup_worker():
            while True:
                try:
                    current_time = time.time()
                    files_to_remove = []

                    for file_path, timestamp in self.file_tracker.items():
                        if current_time - timestamp > self.CLEANUP_TIMEOUT:
                            files_to_remove.append(file_path)

                    for file_path in files_to_remove:
                        self._safe_remove_file(Path(file_path))
                        del self.file_tracker[file_path]

                    time.sleep(60)  # Check every minute
                except Exception as e:
                    print(f"Cleanup thread error: {e}")
                    time.sleep(60)

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _safe_remove_file(self, file_path: Path):
        """Safely remove a file."""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Failed to remove file {file_path}: {e}")

    def validate_file(
        self, file_content: bytes, filename: str, content_type: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Validate uploaded file for security and type compliance.

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check file size
        if len(file_content) > self.MAX_FILE_SIZE:
            return (
                False,
                f"File too large. Maximum size is {self.MAX_FILE_SIZE // (1024*1024)}MB",
            )

        # Check file extension
        file_ext = Path(filename).suffix.lower()

        if file_ext in self.ALLOWED_IMAGE_EXTENSIONS:
            # Validate image file
            if content_type and content_type not in self.ALLOWED_IMAGE_TYPES:
                return False, f"Invalid image type: {content_type}"

            # Check for basic image file signatures
            if not self._validate_image_signature(file_content):
                return False, "File does not appear to be a valid image"

        elif file_ext in self.ALLOWED_BIN_EXTENSIONS:
            # Binary files are allowed for decryption
            pass
        else:
            return False, f"Unsupported file type: {file_ext}"

        return True, ""

    def _validate_image_signature(self, file_content: bytes) -> bool:
        """Validate image file signatures."""
        # JPEG signatures
        if file_content.startswith(b"\xff\xd8\xff"):
            return True
        # PNG signature
        if file_content.startswith(b"\x89PNG\r\n\x1a\n"):
            return True
        # AVIF signature (ftyp box with avif)
        if b"ftypavif" in file_content[:32]:
            return True

        return False

    def save_upload(
        self, file_content: bytes, filename: str, session_id: Optional[str] = None
    ) -> tuple[Path, str]:
        """
        Save uploaded file securely.

        Returns:
            tuple: (file_path, session_id)
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create secure filename
        file_ext = Path(filename).suffix.lower()
        secure_filename = f"{session_id}_upload{file_ext}"
        file_path = self.upload_dir / secure_filename

        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)

        # Track for cleanup
        self.file_tracker[str(file_path)] = time.time()

        return file_path, session_id

    def save_output(self, file_content: bytes, filename: str) -> Path:
        """Save output file."""
        file_path = self.output_dir / filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        # Track for cleanup
        self.file_tracker[str(file_path)] = time.time()

        return file_path

    def get_output_path(self, filename: str) -> Path:
        """Get path for output file."""
        return self.output_dir / filename

    def cleanup_session_files(self, session_id: str):
        """Clean up all files for a session."""
        for file_path in list(self.file_tracker.keys()):
            if session_id in Path(file_path).name:
                self._safe_remove_file(Path(file_path))
                if file_path in self.file_tracker:
                    del self.file_tracker[file_path]

    def schedule_cleanup(self, file_path: Path, delay_seconds: int = 300):
        """Schedule file for cleanup after delay."""

        def delayed_cleanup():
            time.sleep(delay_seconds)
            self._safe_remove_file(file_path)
            if str(file_path) in self.file_tracker:
                del self.file_tracker[str(file_path)]

        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()
