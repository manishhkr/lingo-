# from flask import Flask, jsonify, request
# from qdrant_client import QdrantClient
# from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
# import os
# from urllib.parse import unquote
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # Qdrant configuration
# QDRANT_URL = "https://5bd57eec-5901-44bc-bf2b-6ecec9484d55.us-west-1-0.aws.cloud.qdrant.io:6333"
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.WG6TTng7xH1MZe65n3v5IrYwApHU4e6wo8iJAWV914M"

# # Initialize Qdrant client
# try:
#     client = QdrantClient(
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#         timeout=30
#     )
#     logger.info("Successfully connected to Qdrant")
# except Exception as e:
#     logger.error(f"Failed to connect to Qdrant: {e}")
#     client = None

# @app.route('/query/<path:query_text>', methods=['GET'])
# def query_qdrant(query_text):
#     """
#     Query Qdrant for relevant chunks based on the input text
#     """
#     try:
#         # Decode URL-encoded query text
#         decoded_query = unquote(query_text)
#         logger.info(f"Processing query: {decoded_query}")
        
#         if not client:
#             return jsonify({
#                 "error": "Qdrant client not initialized",
#                 "status": "error"
#             }), 500
        
#         # Get query parameters
#         collection_name = request.args.get('collection', 'default_collection')
#         limit = int(request.args.get('limit', 10))
#         score_threshold = float(request.args.get('threshold', 0.7))
        
#         try:
#             collections = client.get_collections()
#             logger.info(f"Available collections: {[c.name for c in collections.collections]}")
            
#             if collection_name not in [c.name for c in collections.collections]:
#                 # Try to use the first available collection
#                 if collections.collections:
#                     collection_name = collections.collections[0].name
#                     logger.info(f"Using collection: {collection_name}")
#                 else:
#                     return jsonify({
#                         "error": "No collections found in Qdrant",
#                         "status": "error"
#                     }), 404
            
#             # Get collection info to understand the structure
#             collection_info = client.get_collection(collection_name)
#             logger.info(f"Collection info: {collection_info}")
            
#             # Try to search using scroll (get all points and filter by text similarity)
#             # This is a basic approach - ideally you'd use vector search
#             scroll_result = client.scroll(
#                 collection_name=collection_name,
#                 limit=limit,
#                 with_payload=True,
#                 with_vectors=False
#             )
            
#             # Filter results based on text similarity (basic keyword matching)
#             relevant_chunks = []
#             query_words = decoded_query.lower().split()
            
#             for point in scroll_result[0]:  # scroll_result is a tuple (points, next_page_offset)
#                 payload = point.payload or {}
                
#                 # Check if any payload field contains query keywords
#                 relevance_score = 0
#                 matched_content = ""
                
#                 for key, value in payload.items():
#                     if isinstance(value, str):
#                         value_lower = value.lower()
#                         matches = sum(1 for word in query_words if word in value_lower)
#                         if matches > 0:
#                             relevance_score += matches / len(query_words)
#                             if len(matched_content) < len(value):
#                                 matched_content = value
                
#                 if relevance_score > score_threshold:
#                     relevant_chunks.append({
#                         "id": str(point.id),
#                         "score": relevance_score,
#                         "content": matched_content,
#                         "payload": payload
#                     })
            
#             # Sort by relevance score
#             relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
#             relevant_chunks = relevant_chunks[:limit]
            
#             return jsonify({
#                 "query": decoded_query,
#                 "collection": collection_name,
#                 "results": relevant_chunks,
#                 "total_found": len(relevant_chunks),
#                 "status": "success"
#             })
            
#         except Exception as search_error:
#             logger.error(f"Search error: {search_error}")
#             return jsonify({
#                 "error": f"Search failed: {str(search_error)}",
#                 "query": decoded_query,
#                 "status": "error"
#             }), 500
            
#     except Exception as e:
#         logger.error(f"Query processing error: {e}")
#         return jsonify({
#             "error": f"Failed to process query: {str(e)}",
#             "status": "error"
#         }), 500

# @app.route('/collections', methods=['GET'])
# def list_collections():
#     """
#     List all available collections in Qdrant
#     """
#     try:
#         if not client:
#             return jsonify({
#                 "error": "Qdrant client not initialized",
#                 "status": "error"
#             }), 500
        
#         collections = client.get_collections()
#         collection_list = []
        
#         for collection in collections.collections:
#             try:
#                 info = client.get_collection(collection.name)
#                 collection_list.append({
#                     "name": collection.name,
#                     "points_count": info.points_count,
#                     "status": info.status
#                 })
#             except Exception as e:
#                 logger.warning(f"Could not get info for collection {collection.name}: {e}")
#                 collection_list.append({
#                     "name": collection.name,
#                     "points_count": "unknown",
#                     "status": "unknown"
#                 })
        
#         return jsonify({
#             "collections": collection_list,
#             "total": len(collection_list),
#             "status": "success"
#         })
        
#     except Exception as e:
#         logger.error(f"Failed to list collections: {e}")
#         return jsonify({
#             "error": f"Failed to list collections: {str(e)}",
#             "status": "error"
#         }), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """
#     Health check endpoint
#     """
#     try:
#         if not client:
#             return jsonify({
#                 "status": "unhealthy",
#                 "qdrant": "disconnected"
#             }), 500
        
#         # Try to get collections as a health check
#         collections = client.get_collections()
        
#         return jsonify({
#             "status": "healthy",
#             "qdrant": "connected",
#             "collections_count": len(collections.collections)
#         })
        
#     except Exception as e:
#         return jsonify({
#             "status": "unhealthy",
#             "error": str(e)
#         }), 500

# @app.route('/', methods=['GET'])
# def root():
#     """
#     Root endpoint with API documentation
#     """
#     return jsonify({
#         "message": "Qdrant Query Service",
#         "endpoints": {
#             "/query/<query_text>": "Search for relevant chunks",
#             "/collections": "List available collections",
#             "/health": "Health check"
#         },
#         "example": "GET /query/what%20is%20lingo?collection=your_collection&limit=5&threshold=0.1",
#         "parameters": {
#             "collection": "Collection name (optional)",
#             "limit": "Maximum results to return (default: 10)",
#             "threshold": "Minimum relevance score (default: 0.0)"
#         }
#     })

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({
#         "error": "Endpoint not found",
#         "status": "error"
#     }), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({
#         "error": "Internal server error",
#         "status": "error"
#     }), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)




from flask import Flask, jsonify, request
from pinecone import Pinecone
import os
from urllib.parse import unquote
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Pinecone configuration
PINECONE_API_KEY = "pcsk_4uTXQQ_2k4G6tD58zQCVmyL5gqQf21FdKVWTagJKzNANba3jyhaVKASR1WwsYHBktriedt"
PINECONE_ENVIRONMENT = "us-east-1"
DEFAULT_INDEX_NAME = "my-documents"

# Initialize Pinecone client and embeddings
pc = None
embeddings = None

def initialize_pinecone():
    """Initialize Pinecone client"""
    global pc
    try:
        if pc is None:
            logger.info("Initializing Pinecone client...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("Successfully connected to Pinecone")
        return pc
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        return None

def initialize_embeddings():
    """Initialize embeddings model"""
    global embeddings
    try:
        if embeddings is None:
            logger.info("Initializing embeddings model...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Successfully initialized embeddings model")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        return None

def generate_query_embedding(text):
    """Generate embedding for query text"""
    try:
        emb = initialize_embeddings()
        if not emb:
            return None, "Embeddings model not initialized"
        
        if not text or not text.strip():
            return None, "Empty query text"
        
        embedding = emb.embed_query(text.strip())
        
        if not isinstance(embedding, list) or len(embedding) != 384:
            return None, f"Invalid embedding format or dimension: {len(embedding) if isinstance(embedding, list) else 'N/A'}"
        
        if not all(isinstance(x, (int, float)) and not (np.isnan(x) or np.isinf(x)) for x in embedding):
            return None, "Embedding contains invalid values"
        
        return embedding, None
        
    except Exception as e:
        return None, str(e)

@app.route('/query/<path:query_text>', methods=['GET'])
def query_pinecone(query_text):
    """
    Query Pinecone for relevant chunks based on the input text
    """
    try:
        # Decode URL-encoded query text
        decoded_query = unquote(query_text)
        logger.info(f"Processing query: {decoded_query}")
        
        client = initialize_pinecone()
        if not client:
            return jsonify({
                "error": "Pinecone client not initialized",
                "status": "error"
            }), 500
        
        # Get query parameters
        index_name = request.args.get('index', DEFAULT_INDEX_NAME)
        limit = int(request.args.get('limit', 10))
        score_threshold = float(request.args.get('threshold', 0.0))
        
        try:
            # Check if index exists
            indexes = client.list_indexes()
            available_indexes = [idx.name for idx in indexes]
            logger.info(f"Available indexes: {available_indexes}")
            
            if index_name not in available_indexes:
                if available_indexes:
                    index_name = available_indexes[0]
                    logger.info(f"Using available index: {index_name}")
                else:
                    return jsonify({
                        "error": "No indexes found in Pinecone",
                        "status": "error"
                    }), 404
            
            # Get the index
            index = client.Index(index_name)
            
            # Get index stats
            stats = index.describe_index_stats()
            logger.info(f"Index '{index_name}' stats: {stats}")
            
            if stats.total_vector_count == 0:
                return jsonify({
                    "query": decoded_query,
                    "index": index_name,
                    "results": [],
                    "total_found": 0,
                    "message": "Index is empty - no vectors to search",
                    "status": "success"
                })
            
            # Generate embedding for the query
            query_embedding, error = generate_query_embedding(decoded_query)
            if error:
                return jsonify({
                    "error": f"Failed to generate query embedding: {error}",
                    "query": decoded_query,
                    "status": "error"
                }), 500
            
            # Perform vector search
            search_results = []
            
            try:
                # Try high precision search first (with filter)
                results_high = index.query(
                    vector=query_embedding,
                    top_k=limit,
                    include_metadata=True,
                    filter={"embedding_verified": True}
                )
                search_results.extend(results_high.matches)
                logger.info(f"High precision search: {len(results_high.matches)} results")
            except Exception as e:
                logger.warning(f"High precision search failed: {e}")
            
            # If not enough results, try broader search
            if len(search_results) < limit:
                try:
                    results_broad = index.query(
                        vector=query_embedding,
                        top_k=limit * 2,
                        include_metadata=True
                    )
                    
                    # Add results that aren't already included
                    existing_ids = {r.id for r in search_results}
                    for result in results_broad.matches:
                        if result.id not in existing_ids and len(search_results) < limit:
                            search_results.append(result)
                    
                    logger.info(f"Broad search added results: total now {len(search_results)}")
                except Exception as e:
                    logger.warning(f"Broad search failed: {e}")
            
            # Filter by score threshold and format results
            relevant_chunks = []
            for result in search_results:
                if result.score >= score_threshold:
                    metadata = result.metadata or {}
                    relevant_chunks.append({
                        "id": str(result.id),
                        "score": float(result.score),
                        "content": metadata.get("text", ""),
                        "source_file": metadata.get("source_file", "unknown"),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "chunk_size": metadata.get("chunk_size", 0),
                        "payload": metadata
                    })
            
            # Sort by relevance score
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            relevant_chunks = relevant_chunks[:limit]
            
            return jsonify({
                "query": decoded_query,
                "index": index_name,
                "results": relevant_chunks,
                "total_found": len(relevant_chunks),
                "index_stats": {
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension
                },
                "status": "success"
            })
            
        except Exception as search_error:
            logger.error(f"Search error: {search_error}")
            return jsonify({
                "error": f"Search failed: {str(search_error)}",
                "query": decoded_query,
                "status": "error"
            }), 500
            
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return jsonify({
            "error": f"Failed to process query: {str(e)}",
            "status": "error"
        }), 500

@app.route('/indexes', methods=['GET'])
def list_indexes():
    """
    List all available indexes in Pinecone
    """
    try:
        client = initialize_pinecone()
        if not client:
            return jsonify({
                "error": "Pinecone client not initialized",
                "status": "error"
            }), 500
        
        indexes = client.list_indexes()
        index_list = []
        
        for index_info in indexes:
            try:
                index = client.Index(index_info.name)
                stats = index.describe_index_stats()
                index_list.append({
                    "name": index_info.name,
                    "dimension": stats.dimension,
                    "metric": index_info.metric,
                    "points_count": stats.total_vector_count,
                    "status": "ready" if stats.total_vector_count > 0 else "empty"
                })
            except Exception as e:
                logger.warning(f"Could not get stats for index {index_info.name}: {e}")
                index_list.append({
                    "name": index_info.name,
                    "dimension": getattr(index_info, 'dimension', 'unknown'),
                    "metric": getattr(index_info, 'metric', 'unknown'),
                    "points_count": "unknown",
                    "status": "unknown"
                })
        
        return jsonify({
            "indexes": index_list,
            "total": len(index_list),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")
        return jsonify({
            "error": f"Failed to list indexes: {str(e)}",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    try:
        client = initialize_pinecone()
        emb = initialize_embeddings()
        
        if not client:
            return jsonify({
                "status": "unhealthy",
                "pinecone": "disconnected",
                "embeddings": "ready" if emb else "not_ready"
            }), 500
        
        # Try to get indexes as a health check
        indexes = client.list_indexes()
        
        # Test embedding generation
        embedding_test = None
        if emb:
            try:
                test_embedding, error = generate_query_embedding("test query")
                embedding_test = "working" if not error else f"error: {error}"
            except Exception as e:
                embedding_test = f"error: {str(e)}"
        
        return jsonify({
            "status": "healthy",
            "pinecone": "connected",
            "embeddings": "ready" if emb else "not_ready",
            "embedding_test": embedding_test,
            "indexes_count": len(indexes),
            "available_indexes": [idx.name for idx in indexes]
        })
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/debug/<path:query_text>', methods=['GET'])
def debug_query(query_text):
    """
    Debug endpoint to test query processing and embeddings
    """
    try:
        decoded_query = unquote(query_text)
        client = initialize_pinecone()
        emb = initialize_embeddings()
        
        if not client:
            return jsonify({
                "error": "Pinecone client not initialized",
                "status": "error"
            }), 500
        
        index_name = request.args.get('index', DEFAULT_INDEX_NAME)
        
        # Get index info
        try:
            index = client.Index(index_name)
            stats = index.describe_index_stats()
        except Exception as e:
            return jsonify({
                "error": f"Could not access index '{index_name}': {str(e)}",
                "available_indexes": [idx.name for idx in client.list_indexes()]
            }), 500
        
        # Test embedding generation
        query_embedding, embedding_error = generate_query_embedding(decoded_query)
        
        # Get sample results if embedding works
        sample_results = []
        if query_embedding and not embedding_error:
            try:
                sample_results = index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True
                ).matches
            except Exception as e:
                logger.warning(f"Sample query failed: {e}")
        
        formatted_results = [{
            "id": r.id,
            "score": r.score,
            "metadata": r.metadata
        } for r in sample_results]

        return jsonify({
            "query": decoded_query,
            "embedding_error": embedding_error,
            "embedding_length": len(query_embedding) if query_embedding else 0,
            "index_stats": {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            },
            "sample_results": formatted_results,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Debug endpoint failed: {e}")
        return jsonify({
            "error": f"Debug failed: {str(e)}",
            "status": "error"
        }), 500


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with API documentation
    """
    client = initialize_pinecone()
    emb = initialize_embeddings()
    
    # Get system status
    system_status = {
        "pinecone_connected": client is not None,
        "embeddings_ready": emb is not None,
        "ready_for_queries": client is not None and emb is not None
    }
    
    # Get available indexes
    available_indexes = []
    if client:
        try:
            indexes = client.list_indexes()
            available_indexes = [idx.name for idx in indexes]
        except Exception as e:
            logger.warning(f"Could not list indexes: {e}")
    
    return jsonify({
        "message": "Pinecone Query Service",
        "description": "Flask API for querying Pinecone vector database with embedding support",
        "version": "1.0.0",
        "system_status": system_status,
        "available_indexes": available_indexes,
        "endpoints": {
            "/query/<query_text>": "Search for relevant chunks using vector similarity",
            "/indexes": "List available Pinecone indexes",
            "/health": "Health check for all system components",
            "/debug/<query_text>": "Debug query processing and embeddings"
        },
        "example_usage": {
            "basic_query": "GET /query/what%20is%20machine%20learning?index=my-documents&limit=5&threshold=0.7",
            "with_parameters": {
                "index": "Specify which Pinecone index to search (default: my-documents)",
                "limit": "Maximum results to return (default: 10)",
                "threshold": "Minimum similarity score (default: 0.0)"
            }
        },
        "features": [
            "Vector similarity search using embeddings",
            "Automatic embedding generation for queries",
            "Multi-level search (filtered + broad)",
            "Comprehensive error handling",
            "Debug capabilities for troubleshooting",
            "Health monitoring for all components"
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": "error",
        "available_endpoints": ["/", "/query/<text>", "/indexes", "/health", "/debug/<text>"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Pinecone Query Service")
    print("=" * 50)
    
    # Initialize components on startup
    print("🔗 Initializing Pinecone client...")
    pc_client = initialize_pinecone()
    
    print("🧠 Initializing embeddings model...")
    emb_model = initialize_embeddings()
    
    if pc_client and emb_model:
        print("✅ All components initialized successfully!")
        print("🎯 Ready to serve vector search queries!")
    else:
        print("⚠️  Some components failed to initialize")
        print("💡 Check logs for specific errors")
    
    print("=" * 50)
    
    app.run(debug=False,use_reloader=False, host='0.0.0.0', port=5001)