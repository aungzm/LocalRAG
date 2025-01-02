import os
import hashlib
import csv
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FolderWatcher(FileSystemEventHandler):
    def __init__(self, folder_to_watch, callback):
        self.folder_to_watch = folder_to_watch
        self.hash_storage = os.path.join(folder_to_watch, "file_hashes.csv")
        self.callback = callback
        self.lock = threading.Lock()
        self.is_initializing = True  # Flag to prevent runtime updates during initialization
        self.file_hashes = self._initialize_hashes()
        self.is_initializing = False  # Allow runtime updates after initialization

    def _initialize_hashes(self):
        """Load or initialize hashes and detect changes."""
        existing_hashes = self._load_hashes()  # Load from file_hashes.csv
        current_hashes = self._compute_all_hashes()  # Compute current folder state

        added_files = set(current_hashes) - set(existing_hashes)
        deleted_files = set(existing_hashes) - set(current_hashes)
        modified_files = {
            file for file in current_hashes
            if file in existing_hashes and current_hashes[file] != existing_hashes[file]
        }

        # Log and process changes
        for file in added_files:
            self.callback("file_added", file)
        for file in deleted_files:
            self.callback("file_deleted", file)
        for file in modified_files:
            self.callback("file_modified", file)

        # Update hashes to reflect current state
        all_hashes = {**existing_hashes, **current_hashes}
        for file in deleted_files:
            all_hashes.pop(file, None)

        self._save_hashes(all_hashes)
        return all_hashes

    def _save_hashes(self, file_hashes):
        """Save hashes to file_hashes.csv."""
        with self.lock:
            try:
                with open(self.hash_storage, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for file_path, file_hash in file_hashes.items():
                        writer.writerow([file_path, file_hash])
            except Exception as e:
                print(f"Error saving hashes: {e}")

    def _load_hashes(self):
        if not os.path.exists(self.hash_storage):
            return {}

        hashes = {}
        with self.lock:
            try:
                with open(self.hash_storage, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) == 2:
                            file_path, file_hash = row
                            normalized_path = os.path.abspath(file_path)
                            # Remove this check or handle it differently:
                            # if os.path.exists(normalized_path):
                            hashes[normalized_path] = file_hash
            except Exception as e:
                print(f"Error loading hashes: {e}")
        return hashes

    def _compute_all_hashes(self):
        """Compute hashes for all files in the folder with normalized paths."""
        hashes = {}
        for root, _, files in os.walk(self.folder_to_watch):
            for file in files:
                if file == "file_hashes.csv":  # Skip the hash storage file
                    continue
                file_path = os.path.abspath(os.path.join(root, file))  # Normalize path
                hashes[file_path] = self._compute_file_hash(file_path)
        return hashes

    def _compute_file_hash(self, file_path):
        """Compute a SHA256 hash for a single file."""
        if not os.path.isfile(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def on_modified(self, event):
        self._handle_event(event)

    def on_created(self, event):
        self._handle_event(event)

    def on_deleted(self, event):
        self._handle_event(event)

    def _handle_event(self, event):
        """Handle file events during runtime."""
        if self.is_initializing:  # Skip runtime updates during initialization
            return

        if event.is_directory:
            return

        file_path = os.path.abspath(event.src_path)

        # Skip file_hashes.csv, .tmp and ~$ prefixed files
        if file_path.endswith("file_hashes.csv") or file_path.endswith(".tmp") or os.path.basename(file_path).startswith("~$"):
            return

        if event.event_type == "file_deleted":
            if file_path in self.file_hashes:
                del self.file_hashes[file_path]
                self._save_hashes(self.file_hashes)
                self.callback("deleted", file_path)
            return

        # Handle created or modified files
        new_hash = self._compute_file_hash(file_path)
        old_hash = self.file_hashes.get(file_path)

        if old_hash is None and new_hash:  # New file added
            self.file_hashes[file_path] = new_hash
            self._save_hashes(self.file_hashes)
            self.callback("file_added", file_path)
        elif old_hash != new_hash and new_hash:  # File modified
            self.file_hashes[file_path] = new_hash
            self._save_hashes(self.file_hashes)
            self.callback("file_modified", file_path)


def on_file_change(event_type, file_path):
    print(f"File {event_type}: {file_path}")

if __name__ == "__main__":
    folder_to_watch = "./watched_folder"
    if not os.path.exists(folder_to_watch):
        os.makedirs(folder_to_watch)

    event_handler = FolderWatcher(folder_to_watch, on_file_change)
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=True)

    print(f"Watching folder: {folder_to_watch}")
    try:
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
