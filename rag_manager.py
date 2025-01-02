from typing import Dict, Callable
from langchain_ollama import OllamaLLM as Ollama  # local LLM
from langchain_community.chat_models import ChatOpenAI        # API-based LLM (OpenAI)
from langchain.schema import HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# Embeddings Map
EMBEDDINGS_MAP: Dict[str, Callable[[], object]] = {
    # If the user picks "ollama", we’ll use OllamaEmbeddings
    "ollama": lambda: OllamaEmbeddings(model="nomic-embed-text"),

    # If the user picks "openai", we could use Bedrock or an official OpenAI embedding
    "openai": lambda: BedrockEmbeddings(model="bedrock-embed-text"),
}

def select_embeddings(model_type: str):
    """
    Select the embedding function based on the model_type.
    """
    if model_type not in EMBEDDINGS_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available options: {list(EMBEDDINGS_MAP.keys())}"
        )
    return EMBEDDINGS_MAP[model_type]()

def select_llm(model_type: str, model_param: str):
    """
    Return an LLM instance based on the model_type.
    
    """
    if model_type == "ollama":
        # 'model_param' = local model name, e.g. "mistral"
        return Ollama(model=model_param)
    elif model_type == "openai":
        # 'model_param' = the user's API key
        if not model_param:
            raise ValueError("Must provide an OpenAI API key for 'model_param' when model_type='openai'.")
        return ChatOpenAI(
            openai_api_key=model_param,
            model_name="gpt-3.5-turbo",  # Or your default openai model name
            temperature=0
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the user's question appropriately.

Question: {question}

Note: Use the context provided below ONLY if it is relevant. Otherwise, respond based on your general knowledge.

Context:
{context}
"""

NEED_CONTEXT = """
You are a helpful assistant. Analyze the user's question and respond with either "True" or "False" on whether you need more context to answer the question.

Question: {question}
"""


def query_rag(
    query_text: str,
    chroma_path: str,
    model_type: str,  # "openai" or "ollama"
    model_param: str
):
    try:
        # Determine if context is needed
        need_context_template = ChatPromptTemplate.from_template(NEED_CONTEXT)
        llm = select_llm(model_type=model_type, model_param=model_param)

        # Check if more context is needed
        need_context_prompt = need_context_template.format(question=query_text)

        if model_type == "ollama":
            need_context_response = llm.invoke(need_context_prompt)
        elif model_type == "openai":
            response = llm([HumanMessage(content=need_context_prompt)])
            need_context_response = response.content
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        needs_context = False if "false" in need_context_response.strip().lower() else True

        # Initialize context
        context_text = "No relevant context found."

        if needs_context:
            # Select embeddings
            embedding_function = select_embeddings(model_type)
            # Load Chroma vector store
            db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

            # Perform similarity search
            results = db.similarity_search_with_score(query_text, k=5)
            if results:
                # Build context from results
                context_text_list = []
                for doc, score in results:
                    chunk_with_score = f"{doc.page_content}\n(Score: {score:.4f})"
                    context_text_list.append(chunk_with_score)
                context_text = "\n\n---\n\n".join(context_text_list)

        # Build the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt_str = prompt_template.format(context=context_text, question=query_text)

        # Generate response
        if model_type == "ollama":
            response_text = llm.invoke(prompt_str)
        elif model_type == "openai":
            response = llm([HumanMessage(content=prompt_str)])
            response_text = response.content
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Gather sources if context was used
        sources = []
        if needs_context and results:
            sources = [doc.metadata.get("id", None) if doc.metadata else "Unknown" for doc, _ in results]

        formatted_response = f"Response:\n{response_text}"
        if sources:
            formatted_response += f"\n\nSources: {sources}"

        return formatted_response

    except Exception as e:
        print(f"Error during query_rag execution: {e}")
        return "An error occurred while processing your query."
