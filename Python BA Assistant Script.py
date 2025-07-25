import os
import pandas as pd
from dotenv import load_dotenv
# The whisper import is no longer needed.
# We will now use the assemblyai library.
import assemblyai as aai

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough


# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# API Keys and Columns are loaded from the .env file.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# File paths will be requested from the user at runtime.
# Load columns from .env, split the string into a list, and provide a default.
DEFAULT_COLUMNS = [
    "Epic ID", "Epic Name", "User Story ID", "User Story Name", 
    "As a (User Type)", "I want (Action)", "So that (Benefit)", 
    "Acceptance Criteria", "Priority"
]
BACKLOG_COLUMNS_STR = os.getenv("BACKLOG_COLUMNS")
if BACKLOG_COLUMNS_STR:
    BACKLOG_COLUMNS = [col.strip() for col in BACKLOG_COLUMNS_STR.split(',')]
else:
    BACKLOG_COLUMNS = DEFAULT_COLUMNS

# --- Pydantic Data Models for Structured Output ---
class BacklogItem(BaseModel):
    epic_id: str = Field(description="A placeholder ID for the epic, e.g., E01")
    epic_name: str = Field(description="A high-level epic name for the features discussed")
    user_story_id: str = Field(description="A placeholder ID for the user story, e.g., US001")
    user_story_name: str = Field(description="A short, descriptive name for the user story/feature")
    as_a_user_type: str = Field(description="The user persona, e.g., Manager, User, Admin")
    i_want_action: str = Field(description="The specific feature or capability the user wants")
    so_that_benefit: str = Field(description="The value or reason for the feature")
    acceptance_criteria: list[str] = Field(description="A list of 2-3 specific, testable conditions for completion")
    priority: str = Field(description="The estimated priority (High, Medium, Low) based on the client's tone")

class ProductBacklog(BaseModel):
    backlog_items: list[BacklogItem]


def transcribe_audio_with_assemblyai(file_path):
    """
    Transcribes an audio file using the AssemblyAI API.
    """
    if not ASSEMBLYAI_API_KEY:
        print("--- ERROR: ASSEMBLYAI_API_KEY not found in .env file. ---")
        return None
        
    if not os.path.exists(file_path):
        print(f"--- ERROR: Audio file not found at the specified path: {file_path}")
        return None

    print(f"Uploading {file_path} to AssemblyAI for transcription...")
    try:
        aai.settings.api_key = ASSEMBLYAI_API_KEY
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


def create_backlog_with_langchain_rag(docs: list):
    """
    Uses a LangChain RAG pipeline to analyze a transcript and extract structured backlog items.
    """
    print("Initializing LangChain RAG pipeline...")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
    structured_llm = llm.with_structured_output(ProductBacklog)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert Business Analyst. Your task is to analyze the provided context from a meeting transcript and extract ALL requirements, converting them into a structured product backlog.
        
        Analyze the following context carefully and generate a comprehensive list of all user stories mentioned.

        Context:
        {context}
        
        Question: {input}
        """
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    
    vector_store = FAISS.from_documents(splits, embeddings)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Manually construct the RAG chain using LCEL for better control with structured output
    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["input"]) | retriever
        )
        | prompt
        | structured_llm
    )
    
    print("Invoking RAG chain to analyze transcript...")
    # The input dictionary is passed to the start of the chain
    product_backlog_object = rag_chain.invoke({"input": "Find all requirements in the document"})
    
    print("Successfully extracted and structured requirements.")
    
    # The result of the chain is the Pydantic object directly
    if product_backlog_object:
        return [item.dict() for item in product_backlog_object.backlog_items]
    
    return None


def write_to_excel(data, columns, output_path):
    """
    Writes the structured data to an Excel file using pandas.
    """
    if not data:
        print("No data to write to Excel.")
        return
        
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Writing data to {output_path}...")
    
    df = pd.DataFrame(data)

    column_mapping = {
        "epic_id": "Epic ID",
        "epic_name": "Epic Name",
        "user_story_id": "User Story ID",
        "user_story_name": "User Story Name",
        "as_a_user_type": "As a (User Type)",
        "i_want_action": "I want (Action)",
        "so_that_benefit": "So that (Benefit)",
        "acceptance_criteria": "Acceptance Criteria",
        "priority": "Priority"
    }
    df = df.rename(columns=column_mapping)

    df = df[columns]

    df.to_excel(output_path, index=False)
    print(f"Excel file created successfully at: {os.path.abspath(output_path)}")

def main():
    """
    Main function to run the entire pipeline.
    """
    print("--- Starting LangChain RAG BA Assistant Pipeline ---")
    
    if not GEMINI_API_KEY:
        print("--- ERROR: GEMINI_API_KEY not found in .env file. ---")
        return
        
    while True:
        audio_filename = input("Enter the name of your audio file (e.g., meeting.mp3): ")
        if audio_filename:
            break
        print("Audio file name cannot be empty. Please try again.")

    while True:
        excel_filename = input("Enter the desired name for the output Excel file (e.g., backlog.xlsx): ")
        if excel_filename:
            if not excel_filename.endswith('.xlsx'):
                excel_filename += '.xlsx'
            break
        print("Excel file name cannot be empty. Please try again.")
        
    audio_file_path = os.path.join("audios", audio_filename)
    output_excel_path = os.path.join("excels", excel_filename)

    # --- Step 1: Transcription using AssemblyAI ---
    docs = transcribe_audio_with_assemblyai(audio_file_path)

    if not docs:
        print("Could not obtain transcript from audio file. Exiting.")
        return

    # --- Step 2: Information Extraction with LangChain RAG ---
    backlog_data = create_backlog_with_langchain_rag(docs)

    # --- Step 3: Write to Excel ---
    if backlog_data:
        write_to_excel(backlog_data, BACKLOG_COLUMNS, output_excel_path)
    
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
