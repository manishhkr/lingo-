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
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
import os
from urllib.parse import unquote
import logging
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Qdrant configuration
QDRANT_URL = "https://5bd57eec-5901-44bc-bf2b-6ecec9484d55.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.WG6TTng7xH1MZe65n3v5IrYwApHU4e6wo8iJAWV914M"

# Local embedding model - no API keys needed
EMBEDDING_MODEL = None

# Initialize Qdrant client and embedding model
try:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )
    logger.info("Successfully connected to Qdrant")
    
    # Initialize embedding model at startup
    initialize_embedding_model()
    
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    client = None

def initialize_embedding_model():
    """
    Initialize the local embedding model (done once)
    """
    global EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use a lightweight, fast model
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Local embedding model loaded successfully")
        return True
        
    except ImportError:
        logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return False

def get_embedding_local(text):
    """
    Get embedding using local sentence-transformers model
    """
    global EMBEDDING_MODEL
    
    try:
        # Initialize model if not already done
        if EMBEDDING_MODEL is None:
            if not initialize_embedding_model():
                return None
        
        # Generate embedding
        embedding = EMBEDDING_MODEL.encode(text)
        return embedding.tolist()
        
    except Exception as e:
        logger.error(f"Failed to get local embedding: {e}")
        return None

@app.route('/query/<path:query_text>', methods=['GET'])
def query_qdrant(query_text):
    """
    Query Qdrant for relevant chunks using vector search
    """
    try:
        # Decode URL-encoded query text
        decoded_query = unquote(query_text)
        logger.info(f"Processing query: {decoded_query}")
        
        if not client:
            return jsonify({
                "error": "Qdrant client not initialized",
                "status": "error"
            }), 500
        
        # Get query parameters
        collection_name = request.args.get('collection', 'default_collection')
        limit = int(request.args.get('limit', 10))
        score_threshold = float(request.args.get('threshold', 0.7))  # Higher threshold for vector similarity
        
        try:
            collections = client.get_collections()
            logger.info(f"Available collections: {[c.name for c in collections.collections]}")
            
            if collection_name not in [c.name for c in collections.collections]:
                # Try to use the first available collection
                if collections.collections:
                    collection_name = collections.collections[0].name
                    logger.info(f"Using collection: {collection_name}")
                else:
                    return jsonify({
                        "error": "No collections found in Qdrant",
                        "status": "error"
                    }), 404
            
            # Get collection info
            collection_info = client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} has {collection_info.points_count} points")
            
            # Get query embedding using local model
            query_embedding = get_embedding_local(decoded_query)
            
            if query_embedding is None:
                # Fallback to keyword search if embedding fails
                logger.warning("Embedding generation failed, using keyword search")
                return keyword_search(decoded_query, collection_name, limit, score_threshold)
            
            # Perform vector search
            search_result = client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Set to True if you want to return vectors
            )
            
            # Format results
            results = []
            for scored_point in search_result:
                results.append({
                    "id": str(scored_point.id),
                    "score": float(scored_point.score),
                    "payload": scored_point.payload
                })
            
            return jsonify({
                "query": decoded_query,
                "collection": collection_name,
                "results": results,
                "total_found": len(results),
                "search_method": "vector_search",
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

def keyword_search(query_text, collection_name, limit, score_threshold):
    """
    Fallback keyword search method (your original implementation)
    """
    try:
        # Scroll through points and do keyword matching
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=limit * 5,  # Get more points to filter from
            with_payload=True,
            with_vectors=False
        )
        
        # Filter results based on text similarity
        relevant_chunks = []
        query_words = query_text.lower().split()
        
        for point in scroll_result[0]:
            payload = point.payload or {}
            
            # Check if any payload field contains query keywords
            relevance_score = 0
            matched_content = ""
            
            for key, value in payload.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    matches = sum(1 for word in query_words if word in value_lower)
                    if matches > 0:
                        relevance_score += matches / len(query_words)
                        if len(matched_content) < len(value):
                            matched_content = value
            
            if relevance_score > score_threshold:
                relevant_chunks.append({
                    "id": str(point.id),
                    "score": relevance_score,
                    "payload": payload
                })
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        relevant_chunks = relevant_chunks[:limit]
        
        return jsonify({
            "query": query_text,
            "collection": collection_name,
            "results": relevant_chunks,
            "total_found": len(relevant_chunks),
            "search_method": "keyword_search",
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return jsonify({
            "error": f"Keyword search failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/debug/<collection_name>', methods=['GET'])
def debug_collection(collection_name):
    """
    Debug endpoint to understand your collection structure and search issues
    """
    try:
        if not client:
            return jsonify({
                "error": "Qdrant client not initialized",
                "status": "error"
            }), 500
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        
        # Get sample points with vectors to understand structure
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
            with_vectors=True
        )
        
        debug_info = {
            "collection_name": collection_name,
            "total_points": collection_info.points_count,
            "vector_config": {
                "size": getattr(collection_info.config.params.vectors, 'size', 'Not configured'),
                "distance": getattr(collection_info.config.params.vectors, 'distance', 'Not configured')
            },
            "embedding_model_loaded": EMBEDDING_MODEL is not None,
            "sample_points": []
        }
        
        for point in scroll_result[0]:
            point_info = {
                "id": str(point.id),
                "has_vector": point.vector is not None,
                "vector_length": len(point.vector) if point.vector else 0,
                "payload_keys": list(point.payload.keys()) if point.payload else [],
                "payload_sample": {k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                                 for k, v in (point.payload or {}).items()}
            }
            debug_info["sample_points"].append(point_info)
        
        return jsonify({
            "debug_info": debug_info,
            "recommendations": get_debug_recommendations(debug_info),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return jsonify({
            "error": f"Debug failed: {str(e)}",
            "status": "error"
        }), 500

def get_debug_recommendations(debug_info):
    """
    Generate recommendations based on debug info
    """
    recommendations = []
    
    if not debug_info["embedding_model_loaded"]:
        recommendations.append("Install sentence-transformers: pip install sentence-transformers")
    
    if debug_info["total_points"] == 0:
        recommendations.append("Your collection is empty. Add some data first.")
    
    if debug_info["vector_config"]["size"] == "Not configured":
        recommendations.append("Your collection doesn't have vector configuration. You need to create vectors for your data.")
    
    for point in debug_info["sample_points"]:
        if not point["has_vector"]:
            recommendations.append("Your points don't have vectors. You need to add embeddings to your data.")
            break
    
    if debug_info["embedding_model_loaded"] and debug_info["vector_config"]["size"] != "Not configured":
        # Check if dimensions match
        model_dim = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
        stored_dim = debug_info["vector_config"]["size"]
        if stored_dim != model_dim:
            recommendations.append(f"Dimension mismatch: Your stored vectors are {stored_dim}D but the model produces {model_dim}D vectors.")
    
    if not recommendations:
        recommendations.append("Everything looks good! Try lowering the similarity threshold if you're not getting results.")
    
    return recommendations
def test_data(collection_name):
    """
    Test endpoint to see what data is actually in your collection
    """
    try:
        if not client:
            return jsonify({
                "error": "Qdrant client not initialized",
                "status": "error"
            }), 500
        
        # Get a few sample points to see the structure
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        sample_points = []
        for point in scroll_result[0]:
            sample_points.append({
                "id": str(point.id),
                "payload_keys": list(point.payload.keys()) if point.payload else [],
                "payload": point.payload
            })
        
        collection_info = client.get_collection(collection_name)
        
        return jsonify({
            "collection": collection_name,
            "total_points": collection_info.points_count,
            "sample_points": sample_points,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Test data error: {e}")
        return jsonify({
            "error": f"Failed to get test data: {str(e)}",
            "status": "error"
        }), 500

# Keep your existing endpoints
@app.route('/collections', methods=['GET'])
def list_collections():
    """
    List all available collections in Qdrant
    """
    try:
        if not client:
            return jsonify({
                "error": "Qdrant client not initialized",
                "status": "error"
            }), 500
        
        collections = client.get_collections()
        collection_list = []
        
        for collection in collections.collections:
            try:
                info = client.get_collection(collection.name)
                collection_list.append({
                    "name": collection.name,
                    "points_count": info.points_count,
                    "status": info.status,
                    "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else "unknown"
                })
            except Exception as e:
                logger.warning(f"Could not get info for collection {collection.name}: {e}")
                collection_list.append({
                    "name": collection.name,
                    "points_count": "unknown",
                    "status": "unknown",
                    "vector_size": "unknown"
                })
        
        return jsonify({
            "collections": collection_list,
            "total": len(collection_list),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return jsonify({
            "error": f"Failed to list collections: {str(e)}",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    try:
        if not client:
            return jsonify({
                "status": "unhealthy",
                "qdrant": "disconnected"
            }), 500
        
        collections = client.get_collections()
        
        return jsonify({
            "status": "healthy",
            "qdrant": "connected",
            "collections_count": len(collections.collections)
        })
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with API documentation
    """
    return jsonify({
        "message": "Qdrant Query Service with Vector Search",
        "endpoints": {
            "/query/<query_text>": "Search for relevant chunks using vector similarity",
            "/collections": "List available collections",
            "/test_data/<collection_name>": "Get sample data from a collection",
            "/health": "Health check"
        },
        "example": "GET /query/what%20is%20lingo?collection=your_collection&limit=5&threshold=0.7",
        "parameters": {
            "collection": "Collection name (optional)",
            "limit": "Maximum results to return (default: 10)",
            "threshold": "Minimum similarity score (default: 0.7 for vector search, 0.0 for keyword search)"
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": "error"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)