from venv import create
from prisma import Prisma
from folderwatch import FolderWatcher
from watchdog.observers import Observer
from rag_manager import query_rag
from vector_db import run
import os
import asyncio


prisma = Prisma()
class Chat():
    def __init__(self):
        self.id = None
        self.name = ""
        self.watched_folder = ""
        self.llm_type = ""  # ollama or openai
        self.llm_args = ""  # model name or api key
        self.db_path = ""  # path to the chroma vector database
    
    async def new_chat(self, name: str, watched_folder: str, llm_type: str, llm_args: str, db_path: str):
        if None in [name, watched_folder, llm_type, llm_args]:
            raise ValueError("All arguments must be provided.")

        # Ensure Prisma is connected before querying
        if not prisma.is_connected():
            await prisma.connect()
            print("Prisma reconnected in new_chat.")

        self.name = name
        self.watched_folder = watched_folder
        self.llm_type = llm_type
        self.llm_args = llm_args
        self.db_path = db_path
        await self.save_to_db()

        created_chat = await prisma.chatinfo.find_first(
            where={
                'chatName': self.name
            }
        )
        if created_chat:
            self.id = created_chat.chatId
            self._finalize_id()
        asyncio.create_task(self.start_watch())  # Run folder watching asynchronously

    
    def _finalize_id(self): # Roundabout way to make id immutable
        if hasattr(self, '_id_finalized') and self._id_finalized:
            raise AttributeError("ID is already finalized and cannot be modified.")
        self._id_finalized = True
    
    def modify_name(self, name: str):
        self.name = name
        self.update_chat_info()
    
    def modify_watched_folder(self, folder: str):
        self.watched_folder = folder
        self.update_chat_info()

    def modify_db_path(self, db_path: str):
        self.db_path = db_path
        run(db_path, self.watched_folder, self.llm_type, "reset") # Reset the database
        self.update_chat_info()
        run(db_path, self.watched_folder, self.llm_type, "full") # Re-scan the folder

    def modify_llm_type_and_args(self, llm_type: str, llm_args: str):
        self.llm_type = llm_type
        self.llm_args = llm_args
        self.update_chat_info()
    
    async def load_chat(self, chat_id: int):
        # Ensure Prisma is connected before querying
        if not prisma.is_connected():
            await prisma.connect()
        chat_info = await prisma.chatinfo.find_first(
            where={
                'chatId': chat_id
            }
        )
        if not chat_info:
            raise ValueError(f"Chat with ID {chat_id} not found.")
        self.id = chat_info.chatId
        self.name = chat_info.chatName
        self.watched_folder = chat_info.watchingFolder
        self.llm_type = chat_info.llmType
        self.llm_args = chat_info.llmArgs
        self.db_path = chat_info.dbPath
        print(f"Chat '{self.name}' loaded successfully.")

    
    async def save_to_db(self):
        await prisma.chatinfo.create(
            data={
                'chatName': self.name,
                'watchingFolder': self.watched_folder,
                'llmType': self.llm_type,
                'llmArgs': self.llm_args,
                'dbPath': self.db_path
            }
        )
    
    def update_chat_info(self):
        prisma.chatinfo.update(
            where={
                'chatId': self.name
            },
            data={
                'chatName': self.name,
                'watchingFolder': self.watched_folder,
                'llmType': self.llm_type,
                'llmArgs': self.llm_args,
                'dbPath': self.db_path
            }
        )
    
    async def start_watch(self):
        if not os.path.exists(self.watched_folder):
            os.makedirs(self.watched_folder)
        run(self.db_path, self.watched_folder, self.llm_type, "full")

        def on_file_change(event_type, file_path):
            print(f"File {event_type}: {file_path}")
            if event_type == "file_added":
                run(self.db_path, self.watched_folder, self.llm_type, "full")
            elif event_type == "file_modified":
                run(self.db_path, file_path, self.llm_type, "modify")
            elif event_type == "deleted":
                run(self.db_path, file_path, self.llm_type, "remove")
            else:
                print(f"Unknown event type: {event_type}")
        event_handler = FolderWatcher(self.watched_folder, on_file_change)
        observer = Observer()
        observer.schedule(event_handler, self.watched_folder, recursive=True)

        print(f"Watching folder: {self.watched_folder}")
        observer.start()

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            observer.stop()
            observer.join()

    async def query(self, query_text: str):
        await prisma.chatlog.create(
            data={
                'chatId': self.id,
                'message': query_text,
                'sender': 'user'
            }
        )
        response_text = query_rag(query_text, self.db_path, self.llm_type, self.llm_args)
        if not response_text:
            response_text = "No relevant information found."
        await prisma.chatlog.create(
            data={
                'chatId': self.id,
                'message': response_text,
                'sender': 'bot'
            }
        )
        return response_text
    
    async def retrieve_chatLogs(self, number_of_logs: int):
        logs = prisma.chatlog.find_many(
            where={
                'chatId': self.id
            },
            order_by=[
                {
                    'createdAt': 'desc'
                }
            ],
            take=number_of_logs
        )
        return logs
    
async def main():
    await prisma.connect()  # Connect to Prisma
    try:
        chat = Chat()
        await chat.new_chat("Test Chat", "./watched_folder", "ollama", "mistral", "./db")
        print("Chat initialized. You can now query the chat. Press Ctrl+C to exit.")

        while True:
            try:
                query_text = input("Enter your query (or type 'exit' to quit): ")
                if query_text.lower() == 'exit':
                    print("Exiting the chat.")
                    break

                response = await chat.query(query_text)
                print(f"Response: {response}")
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Exiting.")
                break
    finally:
        await prisma.disconnect()  # Ensure proper disconnection
        print("Disconnected from Prisma.")


if __name__ == "__main__":
    asyncio.run(main())
