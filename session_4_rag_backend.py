
import os
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get("GEMINI_API_KEY")

"""##**Section 1: Uploading PDF**
In this section, we'll implement the functionality to upload PDF files. For this notebook demonstration, we'll assume the PDF is in a local path.
"""

def upload_pdf(pdf_path):
    """
    Function to handle PDF uploads.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: PDF file path if successful
    """
    try:
        # In a real application with Streamlit, you would use:
        # uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        # But for this notebook, we'll just verify the file exists

        if os.path.exists(pdf_path):
            print(f"PDF file found at: {pdf_path}")
            return pdf_path
        else:
            print(f"Error: File not found at {pdf_path}")
            return None
    except Exception as e:
        print(f"Error uploading PDF: {e}")
        return None

attention_paper_path = "/content/attention_paper.pdf"

upload_pdf(attention_paper_path)

"""##**Section 2: Parsing the PDF and Creating Text Files**
Now we'll extract the text content from the uploaded PDFs.
"""

def parse_pdf(pdf_path):
    """
    Function to extract text from PDF files.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""

        # Using pdfplumber to extract text
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

        # Save the extracted text to a file (optional)
        text_file_path = pdf_path.replace('.pdf', '.txt')
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"PDF parsed successfully, extracted {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None

text_file = parse_pdf(attention_paper_path)

"""##**Section 3: Creating Document Chunks**
To effectively process and retrieve information, we need to break down our document into manageable chunks.
"""

def create_document_chunks(text):
    """
    Function to split the document text into smaller chunks for processing.

    Args:
        text (str): The full text from the PDF

    Returns:
        list: List of text chunks
    """
    try:
        # Initialize the text splitter
        # We can tune these parameters based on our needs and model constraints
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # Size of each chunk in characters
            chunk_overlap=100,      # Overlap between chunks to maintain context
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Hierarchy of separators to use when splitting
        )

        # Split the text into chunks
        chunks = text_splitter.split_text(text)

        print(f"Document split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"Error creating document chunks: {e}")
        return []

text_chunks = create_document_chunks(text_file)

"""##**Section 4: Embedding the Documents**
Now we'll create vector embeddings for each text chunk using Gemini's embedding model.
"""

def embed_and_view(text_chunks):
    """
    Embed document chunks and display their numeric embeddings.

    Args:
        text_chunks (list): List of text chunks from the document
    """
    try:
        # Initialize the Gemini embeddings
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"  # Specify the Gemini Embedding model
        )

        print("Embedding model initialized successfully")

        # Generate and display embeddings for all chunks
        for i, chunk in enumerate(text_chunks):
            embedding = embedding_model.embed_query(chunk)
            print(f"Chunk {i} Embedding:\n{embedding}\n")

    except Exception as e:
        print(f"Error embedding documents: {e}")

# Example usage
sample_chunks = ["This is the first chunk.", "This is the second chunk.", "And this is the third chunk."]
embed_and_view(sample_chunks)

def embed_documents(text_chunks):
    """
    Function to generate embeddings for the text chunks.

    Args:
        text_chunks (list): List of text chunks from the document

    Returns:
        object: Embedding model for further use
    """
    try:
        # Initialize the Gemini embeddings
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"  # Specify the Gemini Embedding model
        )

        print("Embedding model initialized successfully")
        return embedding_model, text_chunks
    except Exception as e:
        print(f"Error embedding documents: {e}")
        return None, None

embedded_documents = embed_documents(text_chunks)

"""##**Section 5: Storing in Vector Database (ChromaDB)**
In this section, we'll store the embedded document chunks in a vector database for efficient semantic search.
"""

def store_embeddings(embedding_model, text_chunks):
    """
    Function to store document embeddings in ChromaDB.

    Args:
        embedding_model: The embedding model to use
        text_chunks (list): List of text chunks to embed and store

    Returns:
        object: Vector store for retrieval
    """
    try:
        # Create a vector store from the documents
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db"  # Directory to persist the database
        )

        # Persist the vector store to disk
        vectorstore.persist()

        print(f"Successfully stored {len(text_chunks)} document chunks in ChromaDB")
        return vectorstore
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return None

chroma_store = store_embeddings(embedded_documents[0],embedded_documents[1])

"""##**Section 6: Embedding User Queries**
When a user submits a query, we need to embed it using the same embedding model to find semantically similar chunks.
"""

def embed_query(query, embedding_model):
    """
    Function to embed the user's query.

    Args:
        query (str): User's question
        embedding_model: The embedding model to use

    Returns:
        list: Embedded query vector
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(query)

        print("Query embedded successfully")
        return query_embedding
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

user_query = "Who are the authors of the Attention paper?"

embedded_query = embed_query(user_query, embedded_documents[0])
print(embedded_query)

"""##**Section 7: Retrieval Process**
Now we'll implement the retrieval component that finds the most relevant document chunks based on the user's query.
"""

def retrieve_relevant_chunks(vectorstore, query, embedding_model, k=3):
    """
    Function to retrieve the most relevant document chunks for a query.

    Args:
        vectorstore: The ChromaDB vector store
        query (str): User's question
        embedding_model: The embedding model
        k (int): Number of chunks to retrieve

    Returns:
        list: List of relevant document chunks
    """
    try:
        # Create a retriever from the vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # Can also use "mmr" for Maximum Marginal Relevance
            search_kwargs={"k": k}     # Number of documents to retrieve
        )

        # Retrieve relevant chunks
        relevant_chunks = retriever.get_relevant_documents(query)

        print(f"Retrieved {len(relevant_chunks)} relevant document chunks")
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []

relevant_chunks = retrieve_relevant_chunks(chroma_store, user_query, embedded_documents[0])

relevant_chunks

def get_context_from_chunks(relevant_chunks, splitter="\n\n---\n\n"):
    """
    Extract page_content from document chunks and join them with a splitter.

    Args:
        relevant_chunks (list): List of document chunks from retriever
        splitter (str): String to use as separator between chunk contents

    Returns:
        str: Combined context from all chunks
    """
    # Extract page_content from each chunk
    chunk_contents = []

    for i, chunk in enumerate(relevant_chunks):
        if hasattr(chunk, 'page_content'):
            # Add a chunk identifier to help with tracing which chunk provided what information
            chunk_text = f"[Chunk {i+1}]: {chunk.page_content}"
            chunk_contents.append(chunk_text)

    # Join all contents with the splitter
    combined_context = splitter.join(chunk_contents)

    return combined_context

context = get_context_from_chunks(relevant_chunks)

context

final_prompt = f"""You are a helpful assistant answering questions based on provided context.

The context is taken from academic papers, and might have formatting issues like spaces missing between words.
Please interpret the content intelligently, separating words properly when they appear joined together.

Use ONLY the following context to answer the question.
If the answer cannot be determined from the context, respond with "I cannot answer this based on the provided context."

Context:
{context}

Question: {user_query}

Answer:"""

final_prompt

def generate_response(prompt, model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.3, top_p=0.95):
    """
    Function to generate a response using the Gemini model.

    Args:
        prompt (str): The prompt for the model

    Returns:
        str: Model's response
    """

    llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.2,  # Lower temperature for more focused answers
            top_p=0.95
        )

    response = llm.invoke(prompt)

    return response.content

generate_response(final_prompt)