- [Ollama](https://ollama.com/)

This project combines a **FolderWatcher** utility with a **Chat Management System** powered by local and cloud-based language models (LLMs). It enables real-time file tracking, database updates, and interaction with LLMs for intelligent query handling and document processing.

---

## Features

### FolderWatcher
- Monitors a specified folder for file changes (additions, modifications, deletions).
- Tracks file states using a persistent hash storage file (`file_hashes.csv`).
- Updates the hash storage immediately after each event.
- Supports various file types such as `.pdf`, `.docx`, `.txt`, and `.md`.

### Chat Management System
- Allows users to create, manage, and delete chats.
- Integrates with local (`Ollama`) and cloud-based (`OpenAI`) LLMs.
- Stores chat metadata and logs using a Prisma-based database.
- Uses a **Retrieval-Augmented Generation (RAG)** pipeline for intelligent query handling, leveraging document embeddings stored in a Chroma vector database.

---

## Project Structure

### Main Components

1. **`folderwatch.py`**
   - Implements the `FolderWatcher` class to monitor file changes.
   - Uses the `watchdog` library to track file events.
   - Handles real-time updates to the `file_hashes.csv` file.

2. **`chat.py`**
   - Provides the `Chat` class for managing chat sessions.
   - Handles LLM integration (Ollama and OpenAI).
   - Supports database interaction for chat metadata and logs.

3. **`main.py`**
   - Acts as the entry point for user interactions.
   - Supports chat creation, selection, and deletion.
   - Manages interaction with the FolderWatcher and Chat system.

4. **`rag_manager.py`**
   - Contains logic for embedding selection and LLM interaction.
   - Supports RAG pipeline for intelligent responses.
   - Uses `LangChain` for embedding and LLM integrations.

5. **`vector_db.py`**
   - Handles document processing and interaction with the Chroma vector database.
   - Supports loading, chunking, and embedding document data.
   - Includes methods for adding, modifying, and removing documents from the database.

---

## Installation

### Prerequisites
- Python 3.8+
- `pip` package manager
- ollama installed and your choice of model (to run locally)


### Install Dependencies
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Setup Prisma ORM (for database management):
   ```bash
   prisma generate
   prisma migrate deploy
   ```

---

## Usage
### For locall LLMs
1. Start Ollama (to run locally)
2. Download a model (ollama pull mistral)

### For OpenAI
1. Get OpenAI Api key

### Start chatting
```bash
python main.py
```
Options:
1. Start a new chat.
2. Continue an existing chat.
3. Delete one or more chats.

**Give inputs as required**

---

## Supported File Types
- `.pdf` (PDF files)
- `.docx` (Microsoft Word files)
- `.pptx` (PowerPoint files)
- `.txt`, `.md` (Text and Markdown files)

---

## Integration

### Local LLM (Ollama)
- Requires `ollama` installed and running locally.
- Supports custom local models.

### Cloud LLM (OpenAI)
- Requires an API key for OpenAI.
- Supports models like `gpt-3.5-turbo`.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Prisma](https://www.prisma.io/)
- [Ollama](https://ollama.com/)
- [Watchdog](https://github.com/gorakhargosh/watchdog)
