import os
from dotenv import load_dotenv
import assemblyai as aai

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
        return [Document(page_content=transcript.text, metadata={"source": file_path})]
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

def create_qa_rag_chain(docs: list):
    """
    Creates a LangChain RAG pipeline for question-answering.
    """
    print("Initializing RAG pipeline for Q&A...")

    # --- Step 1: Text Chunking ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # --- Step 2: Vector Embeddings and Vector Store ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever()

    # --- Step 3: Create the Q&A Chain ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question} 
        
        Context: {context} 
        
        Answer:
        """
    )
    
    # This chain uses LCEL to define the sequence of operations.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("--- RAG Q&A Bot is ready. Ask a question about the document. ---")
    return rag_chain


def main():
    """
    Main function to run the chat application.
    """
    load_dotenv()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("--- ERROR: GEMINI_API_KEY not found in .env file. ---")
        return

    print("--- Starting Chat with Document Application ---")
    
    # --- Get and process the audio file ---
    while True:
        audio_filename = input("Enter the name of the audio file to chat with (e.g., meeting.mp3): ")
        if audio_filename:
            break
        print("Audio file name cannot be empty. Please try again.")
        
    audio_file_path = os.path.join("audios", audio_filename)
    
    docs = transcribe_audio_with_assemblyai(audio_file_path)
    if not docs:
        print("Could not process the audio file. Exiting.")
        return

    # --- Create the RAG chain ---
    qa_chain = create_qa_rag_chain(docs)
    if not qa_chain:
        print("Failed to create the Q&A chain. Exiting.")
        return

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
