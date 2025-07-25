import os
from dotenv import load_dotenv
import assemblyai as aai

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# The location where the processed document database will be stored.
CHROMA_PERSIST_DIRECTORY = "chroma_db"

def transcribe_audio_with_assemblyai(file_path):
    """
    Transcribes an audio file using the AssemblyAI API.
    """
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        print("--- ERROR: ASSEMBLYAI_API_KEY not found in .env file. ---")
        return None
        
    if not os.path.exists(file_path):
        print(f"--- ERROR: Audio file not found at the specified path: {file_path}")
        return None

    print(f"\nUploading {file_path} to AssemblyAI for transcription...")
    try:
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"--- ERROR: Transcription failed: {transcript.error} ---")
            return None

        print("Transcription complete.")
        # The transcript text is stored in this Document object
        return [Document(page_content=transcript.text, metadata={"source": file_path})]
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

def get_rag_chain(backlog_columns: list):
    """
    Creates the core LangChain RAG pipeline for question-answering.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    
    # The system prompt is now aware of the backlog structure.
    system_prompt = f"""You are an expert Business Analyst assistant. Your goal is to answer questions about a meeting transcript.
    You are working towards creating a product backlog with the following columns: {backlog_columns}.
    Use the following pieces of retrieved context from the meeting transcript to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {{question}} 
    
    Context: {{context}} 
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # This chain will be connected to a retriever later.
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    """
    Main function to run the chat application.
    """
    load_dotenv()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("--- ERROR: GEMINI_API_KEY not found in .env file. ---")
        return

    # Load the backlog columns from the .env file to provide context to the chat bot.
    DEFAULT_COLUMNS = [
        "Epic ID", "Epic Name", "User Story ID", "User Story Name", 
        "As a (User Type)", "I want to (Action)", "So that (Benefit)", 
        "Acceptance Criteria", "Priority"
    ]
    BACKLOG_COLUMNS_STR = os.getenv("BACKLOG_COLUMNS")
    if BACKLOG_COLUMNS_STR:
        BACKLOG_COLUMNS = [col.strip() for col in BACKLOG_COLUMNS_STR.split(',')]
    else:
        BACKLOG_COLUMNS = DEFAULT_COLUMNS

    print("--- Starting Chat with Document Application ---")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    vector_store = None

    # Check if a database already exists
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print(f"Loading existing document database from '{CHROMA_PERSIST_DIRECTORY}'...")
        vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        print("No existing database found. Please provide an audio file to process.")
        # Get and process the audio file
        while True:
            audio_filename = input("Enter the name of the audio file to process (e.g., meeting.mp3): ")
            if audio_filename:
                break
            print("Audio file name cannot be empty. Please try again.")
            
        audio_file_path = os.path.join("audios", audio_filename)
        
        docs = transcribe_audio_with_assemblyai(audio_file_path)
        if not docs:
            print("Could not process the audio file. Exiting.")
            return

        # Create and persist the vector store
        print("Splitting document and creating new vector database...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        print(f"Database created and saved to '{CHROMA_PERSIST_DIRECTORY}'.")

    # --- Create the full RAG chain with the retriever ---
    retriever = vector_store.as_retriever()
    # Pass the backlog columns to the chain creation function
    qa_chain = {"context": retriever, "question": RunnablePassthrough()} | get_rag_chain(backlog_columns=BACKLOG_COLUMNS)

    print("\n--- RAG Q&A Bot is ready. Ask a question about the document. ---")
    print("--- (To load a new document, delete the 'chroma_db' folder and restart) ---")

    # --- Main Chat Loop ---
    while True:
        try:
            user_question = input("\nYou: ")
            if user_question.lower() in ["quit", "exit"]:
                print("\n--- Chat Session Ended ---")
                break

            print("AI: Thinking...", end='\r')
            response = qa_chain.invoke(user_question)
            print(f"AI: {response} ")

        except KeyboardInterrupt:
            print("\n\n--- Chat Session Ended by user. ---")
            break
        except Exception as e:
            print(f"\n--- An error occurred: {e} ---")
            break

if __name__ == "__main__":
    main()
