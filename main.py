from prisma import Prisma
import asyncio
from chat import Chat
import subprocess

prisma = Prisma()

async def main():
    await prisma.connect()  # Ensure connection is established
    try:
        option = input("Enter '1' to start a new chat or '2' to continue an existing chat or '3' to delete chat(s): ")
        if option == '1':
            await create_chat()
        elif option == '2':
            await select_chat()
        elif option == '3':
            await delete_chat()
        else:
            print("Invalid option. Please enter '1' or '2'.")
    finally:
        await prisma.disconnect()  # Ensure proper disconnection after all operations
        print("Disconnected from Prisma.")


async def delete_chat():
    prisma_chats = await prisma.chatinfo.find_many()  # Fetch all records
    if not prisma_chats:
        print("No existing chats found.")
        return
    print("Select the chat number(s) to delete (use comma between numbers):")
    for idx, chat in enumerate(prisma_chats, start=1):
        print(f"{idx}: {chat.chatName}")
    try:
        chat_numbers = input("Enter the chat number(s) to delete (or press Enter to cancel): ")
        if not chat_numbers.strip():
            print("Cancelled deletion.")
            return
        chat_numbers = [int(num) for num in chat_numbers.split(",")]  # Convert input to a list of integers
        for chat_number in chat_numbers:
            if 1 <= chat_number <= len(prisma_chats):
                selected_chat = prisma_chats[chat_number - 1]
                print(f"Deleting chat: {selected_chat.chatName}")
                await prisma.chatinfo.delete(where={"chatId": selected_chat.chatId})
                print("Chat deleted.")
            else:
                print(f"Invalid chat number: {chat_number}")
    except ValueError:
        print("Cancelled deletion.")

async def select_chat():
    prisma_chats = await prisma.chatinfo.find_many()  # Fetch all records
    if not prisma_chats:
        print("No existing chats found.")
        return

    print("Select a chat to continue:")
    for idx, chat in enumerate(prisma_chats, start=1):
        print(f"{idx}: {chat.chatName}")

    try:
        chat_number = input("Enter the chat number to continue (or press Enter to cancel): ")
        if not chat_number.strip():
            print("Cancelled selection.")
            return

        chat_number = int(chat_number)  # Convert input to an integer
        if 1 <= chat_number <= len(prisma_chats):
            selected_chat = prisma_chats[chat_number - 1]
            print(f"Continuing chat: {selected_chat.chatName}")
            chat = Chat()
            await chat.load_chat(selected_chat.chatId)  # Call load_chat while connected
            print("Chat loaded. You can now query the chat. Press Ctrl+C to exit.")
            if chat.llm_type == "ollama":
                print("Starting Ollama server...")
                await subprocess.run(["ollama", "serve"]) # Assumes relevant models are installed because you ran them already
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
        else:
            print("Invalid chat number.")
    except ValueError:
        print("Cancelled selection.")


async def create_chat():
    try:
        chat = Chat()
        chat_name = input("Enter the chat name: ")
        watched_folder = input("Enter the path to the watched folder: ")
        llm_type = input("Enter the model type ('ollama' or 'openai'): ")

        if llm_type == "ollama":
            llm_args = input("Enter the model name for Ollama: ")
        elif llm_type == "openai":
            llm_args = input("Enter the API key for OpenAI: ")
        else:
            print("Invalid model type. Please enter 'ollama' or 'openai'.")
            return
        vector_db_path = input("Enter the path to store the vector database: ")
        if chat.llm_type == "ollama":
            print("Starting Ollama server...")
            subprocess.run(["ollama", "serve"])
            model = llm_args
            result = await subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model not in result.stdout:
                print(f"Model {model} not found. Please download it first")
        await chat.new_chat(chat_name, watched_folder, llm_type, llm_args, vector_db_path)
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
    except Exception as e:
        print(f"Error creating chat: {e}")
        # Reconnect Prisma and report the error
        if not prisma.is_connected():
            print("Attempting to reconnect Prisma...")
            await prisma.connect()
    finally:
        if prisma.is_connected():
            await prisma.disconnect()
        print("Disconnected from Prisma.")

if __name__ == "__main__":
    asyncio.run(main())
