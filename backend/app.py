from flask import Flask, request, jsonify
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import json
from datetime import datetime
from urllib.parse import unquote
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store the vector store
vector_store = None
loaded_files_info = []
embeddings = None

# Configuration
UPLOAD_FOLDER = 'uploads'
VECTOR_DB_PATH = 'vector_db'
ALLOWED_EXTENSIONS = {'txt'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def initialize_embeddings():
    """Initialize embeddings."""
    global embeddings
    try:
        if embeddings is None:
            logger.info("Initializing embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Embeddings initialized successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        return None

def get_files_hash():
    """Get a hash of all files in upload folder to detect changes."""
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            return ""
        
        files_info = []
        for filename in sorted(os.listdir(UPLOAD_FOLDER)):
            if filename.endswith('.txt'):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                stat = os.stat(file_path)
                files_info.append(f"{filename}:{stat.st_size}:{stat.st_mtime}")
        
        return "|".join(files_info)
    except Exception as e:
        logger.error(f"Error getting files hash: {e}")
        return ""

def should_rebuild_vector_db():
    """Check if vector database should be rebuilt based on file changes."""
    try:
        metadata_file = os.path.join(VECTOR_DB_PATH, 'files_metadata.json')
        
        # If no metadata file exists, rebuild
        if not os.path.exists(metadata_file):
            logger.info("No metadata file found, will rebuild vector database")
            return True
        
        # If no vector database files exist, rebuild
        if not os.path.exists(os.path.join(VECTOR_DB_PATH, 'index.faiss')):
            logger.info("No vector database files found, will rebuild")
            return True
        
        # Load existing metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Compare current files hash with stored hash
        current_hash = get_files_hash()
        stored_hash = metadata.get('files_hash', '')
        
        if current_hash != stored_hash:
            logger.info("Files have changed, will rebuild vector database")
            return True
        
        logger.info("Files unchanged, will load existing vector database")
        return False
        
    except Exception as e:
        logger.error(f"Error checking if rebuild needed: {e}")
        return True

def load_files_from_folder():
    """Load and process all text files from the uploads folder."""
    global vector_store, loaded_files_info
    
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            logger.error("Upload folder does not exist")
            return False, "Upload folder does not exist"
        
        # Get all .txt files from uploads folder
        txt_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.txt')]
        
        if not txt_files:
            logger.warning("No .txt files found in uploads folder")
            return False, "No .txt files found in uploads folder"
        
        logger.info(f"Found {len(txt_files)} .txt files: {txt_files}")
        
        # Initialize embeddings
        emb = initialize_embeddings()
        if not emb:
            return False, "Failed to initialize embeddings"
        
        all_documents = []
        loaded_files_info = []
        
        # Process each text file
        for filename in txt_files:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            try:
                logger.info(f"Processing file: {filename}")
                
                # Load the document
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                
                # Add filename to metadata
                for doc in documents:
                    doc.metadata['source_file'] = filename
                
                # Split the text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                file_chunks = text_splitter.split_documents(documents)
                
                # Add to all documents
                all_documents.extend(file_chunks)
                
                # Track file info
                file_stat = os.stat(file_path)
                loaded_files_info.append({
                    "filename": filename,
                    "file_path": file_path,
                    "chunks_count": len(file_chunks),
                    "file_size": file_stat.st_size,
                    "file_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "loaded_time": datetime.now().isoformat()
                })
                
                logger.info(f"Loaded {filename}: {len(file_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        if not all_documents:
            return False, "No documents could be processed"
        
        logger.info(f"Creating vector store from {len(all_documents)} document chunks...")
        
        # Create vector store from all documents
        vector_store = FAISS.from_documents(all_documents, emb)
        
        # Save vector store
        vector_store.save_local(VECTOR_DB_PATH)
        
        # Save files metadata with hash
        metadata = {
            "files_info": loaded_files_info,
            "files_hash": get_files_hash(),
            "created_time": datetime.now().isoformat(),
            "total_files": len(loaded_files_info),
            "total_chunks": len(all_documents)
        }
        
        metadata_file = os.path.join(VECTOR_DB_PATH, 'files_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        total_chunks = len(all_documents)
        total_files = len(loaded_files_info)
        
        logger.info(f"Successfully created vector database with {total_files} files and {total_chunks} chunks")
        return True, {"files_loaded": total_files, "total_chunks": total_chunks}
        
    except Exception as e:
        logger.error(f"Error loading files from folder: {e}")
        return False, str(e)

def load_existing_vector_db():
    """Load existing vector database if it exists and is valid."""
    global vector_store, loaded_files_info
    
    try:
        # Check if vector database files exist
        index_file = os.path.join(VECTOR_DB_PATH, 'index.faiss')
        metadata_file = os.path.join(VECTOR_DB_PATH, 'files_metadata.json')
        
        if not os.path.exists(index_file):
            logger.info("No existing vector database index found")
            return False, "No existing vector database found"
        
        # Initialize embeddings first
        emb = initialize_embeddings()
        if emb is None:
            return False, "Failed to initialize embeddings"
        
        # Load existing vector store
        logger.info("Loading existing vector database...")
        vector_store = FAISS.load_local(VECTOR_DB_PATH, emb, allow_dangerous_deserialization=True)
        
        # Load file info from metadata file
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                loaded_files_info = metadata.get('files_info', [])
        else:
            loaded_files_info = []
        
        logger.info(f"Loaded existing vector database with {len(loaded_files_info)} files")
        return True, f"Loaded existing database with {len(loaded_files_info)} files"
        
    except Exception as e:
        logger.error(f"Error loading existing vector database: {e}")
        return False, str(e)

def auto_load_on_startup():
    """Automatically load vector database on startup - always from upload folder if files exist."""
    try:
        logger.info("=" * 60)
        logger.info("STARTUP: Auto-loading vector database...")
        
        # Check if upload folder has any .txt files
        txt_files = []
        if os.path.exists(UPLOAD_FOLDER):
            txt_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.txt')]
        
        if not txt_files:
            logger.warning("STARTUP: No .txt files found in uploads folder")
            logger.info("STARTUP: Please add .txt files to the uploads folder and restart or call /load endpoint")
            return False
        
        logger.info(f"STARTUP: Found {len(txt_files)} .txt files in uploads folder")
        
        # Check if we should rebuild the vector database
        if should_rebuild_vector_db():
            logger.info("STARTUP: Building vector database from upload folder...")
            success, result = load_files_from_folder()
            if success:
                logger.info(f"STARTUP: Successfully built vector database - {result}")
                return True
            else:
                logger.error(f"STARTUP: Failed to build vector database - {result}")
                return False
        else:
            logger.info("STARTUP: Loading existing vector database...")
            success, message = load_existing_vector_db()
            if success:
                logger.info(f"STARTUP: {message}")
                return True
            else:
                logger.warning(f"STARTUP: Failed to load existing database, rebuilding... - {message}")
                # If loading existing fails, rebuild from files
                success, result = load_files_from_folder()
                if success:
                    logger.info(f"STARTUP: Successfully rebuilt vector database - {result}")
                    return True
                else:
                    logger.error(f"STARTUP: Failed to rebuild vector database - {result}")
                    return False
        
    except Exception as e:
        logger.error(f"STARTUP: Error in auto_load_on_startup: {e}")
        return False
    finally:
        logger.info("=" * 60)

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation."""
    return jsonify({
        "message": "Flask RAG API - Simple Text Response",
        "description": "Place your .txt files in the 'uploads' folder and query them",
        "status": {
            "vector_database_loaded": vector_store is not None,
            "ready_for_queries": vector_store is not None,
            "loaded_files_count": len(loaded_files_info)
        },
        "endpoints": {
            "GET /query/<question>": "Query with URL parameter (e.g., /query/machine learning)",
            "POST /query": "Query with JSON body {'question': 'your question'}",
            "POST /load": "Manually reload all .txt files from uploads folder",
            "GET /status": "Check current status and loaded files",
            "GET /files": "List files in uploads folder",
            "POST /reload": "Force reload files from uploads folder"
        },
        "usage_examples": [
            "http://localhost:5000/query/machine learning",
            "http://localhost:5000/query/what is python?",
            "curl -X POST -H 'Content-Type: application/json' -d '{\"question\":\"machine learning\"}' http://localhost:5000/query"
        ],
        "setup": "1. Place .txt files in 'uploads' folder, 2. Files are loaded automatically on startup, 3. Use /query to ask questions"
    })

@app.route('/load', methods=['POST'])
def load_files():
    """Manually load and process all text files from the uploads folder."""
    try:
        logger.info("Manual load requested...")
        success, result = load_files_from_folder()
        
        if success:
            return jsonify({
                "message": "Files loaded and processed successfully",
                "result": result,
                "loaded_files": loaded_files_info,
                "status": "ready_for_queries"
            }), 200
        else:
            return jsonify({"error": f"Loading failed: {result}"}), 500
            
    except Exception as e:
        logger.error(f"Error in load endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query/<path:question>', methods=['GET'])
def query_database_url(question):
    """Query the vector database using URL parameter and return combined text."""
    # Decode URL-encoded characters
    question = unquote(question)
    return process_query(question)

@app.route('/query', methods=['POST'])
def query_database_json():
    """Query the vector database using JSON body and return combined text."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return "Error: Please provide a 'question' in the request body", 400
        
        question = data['question']
        return process_query(question)
        
    except Exception as e:
        logger.error(f"Error in JSON query endpoint: {e}")
        return f"Error: {str(e)}", 500

def process_query(question):
    """Process the query and return combined relevant text."""
    try:
        if not vector_store:
            logger.warning("Vector store not available, attempting auto-load...")
            if not auto_load_on_startup():
                return "Error: No files loaded and could not auto-load. Please ensure .txt files are in the uploads folder or call /load endpoint.", 400
        
        # Get number of results (default 5, can be adjusted)
        num_results = 5
        
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": num_results})
        retrieved_docs = retriever.get_relevant_documents(question)
        
        if not retrieved_docs:
            return "No relevant information found in the loaded files.", 200
        
        # Combine all relevant text chunks
        combined_text = ""
        sources = set()
        
        for doc in retrieved_docs:
            source_file = doc.metadata.get('source_file', 'unknown')
            sources.add(source_file)
            combined_text += doc.page_content + "\n\n"
        
        # Clean up the combined text
        combined_text = combined_text.strip()
        
        # Add source information at the end
        source_info = f"\n\n[Sources: {', '.join(sorted(sources))}]"
        final_response = combined_text + source_info
        
        logger.info(f"Query processed: '{question}' -> {len(retrieved_docs)} results from {len(sources)} sources")
        return final_response, 200
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error: {str(e)}", 500

@app.route('/status', methods=['GET'])
def get_status():
    """Check the current status."""
    return jsonify({
        "vector_database_loaded": vector_store is not None,
        "ready_for_queries": vector_store is not None,
        "loaded_files": loaded_files_info,
        "total_files": len(loaded_files_info),
        "uploads_folder": os.path.abspath(UPLOAD_FOLDER),
        "vector_db_path": os.path.abspath(VECTOR_DB_PATH),
        "embeddings_initialized": embeddings is not None
    })

@app.route('/files', methods=['GET'])
def list_files():
    """List all .txt files in the uploads folder."""
    try:
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.endswith('.txt'):
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    file_stats = os.stat(file_path)
                    
                    # Check if file is loaded
                    is_loaded = any(f['filename'] == filename for f in loaded_files_info)
                    
                    files.append({
                        "filename": filename,
                        "size_bytes": file_stats.st_size,
                        "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        "is_loaded": is_loaded
                    })
        
        return jsonify({
            "files_in_folder": files,
            "total_files": len(files),
            "loaded_files_count": len(loaded_files_info),
            "folder_path": os.path.abspath(UPLOAD_FOLDER)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reload', methods=['POST'])
def reload_files():
    """Force reload all files from the uploads folder (ignores existing vector database)."""
    try:
        logger.info("Force reload requested...")
        success, result = load_files_from_folder()
        
        if success:
            return jsonify({
                "message": "Files force reloaded successfully",
                "result": result,
                "loaded_files": loaded_files_info
            }), 200
        else:
            return jsonify({"error": f"Reload failed: {result}"}), 500
            
    except Exception as e:
        logger.error(f"Error in reload endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask RAG API - Simple Text Response")
    print("=" * 60)
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Vector DB folder: {os.path.abspath(VECTOR_DB_PATH)}")
    print("\nSetup Instructions:")
    print("1. Place your .txt files in the 'uploads' folder")
    print("2. Files will be loaded automatically on startup")
    print("3. Query examples:")
    print("   - http://localhost:5000/query/machine learning")
    print("   - http://localhost:5000/query/what is python?")
    print("   - curl -X POST -H 'Content-Type: application/json' \\")
    print("     -d '{\"question\":\"machine learning\"}' http://localhost:5000/query")
    print("=" * 60)
    
    # Create uploads folder with a sample file if it doesn't exist or is empty
    if not os.path.exists(UPLOAD_FOLDER) or not any(f.endswith('.txt') for f in os.listdir(UPLOAD_FOLDER)):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        sample_file = os.path.join(UPLOAD_FOLDER, "sample.txt")
        with open(sample_file, 'w') as f:
            f.write("This is a sample text file for testing the RAG system.\n"
                   "Machine learning is a subset of artificial intelligence that focuses on algorithms.\n"
                   "Python is a popular programming language used for data science and machine learning.\n"
                   "Natural language processing helps computers understand human language.\n"
                   "Deep learning uses neural networks with multiple layers.\n"
                   "Data preprocessing is crucial for machine learning success.\n"
                   "Place your own .txt files in this uploads folder to query your own documents.")
        print(f"Created sample file: {sample_file}")
    
    # Auto-load files on startup - THIS IS THE KEY IMPROVEMENT
    print("\n" + "=" * 60)
    print("STARTUP: Initializing vector database from upload folder...")
    startup_success = auto_load_on_startup()
    
    if startup_success:
        print(f"STARTUP: ✅ Ready to serve queries! Loaded {len(loaded_files_info)} files")
    else:
        print("STARTUP: ⚠️  No files loaded. Add .txt files to uploads folder and restart or call /load")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000)