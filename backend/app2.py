from flask import Flask, jsonify, request
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
import os
from urllib.parse import unquote
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Qdrant configuration
QDRANT_URL = "https://5bd57eec-5901-44bc-bf2b-6ecec9484d55.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.WG6TTng7xH1MZe65n3v5IrYwApHU4e6wo8iJAWV914M"

# Initialize Qdrant client
try:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )
    logger.info("Successfully connected to Qdrant")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    client = None

@app.route('/query/<path:query_text>', methods=['GET'])
def query_qdrant(query_text):
    """
    Query Qdrant for relevant chunks based on the input text
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
        score_threshold = float(request.args.get('threshold', 0.0))
        
        # For text search, you'll need to convert text to embeddings
        # This is a placeholder - you'll need to use your embedding model
        # For now, let's try a search by payload if embeddings aren't available
        
        # Option 1: If you have embeddings for the query
        # query_vector = your_embedding_model.encode(decoded_query)
        
        # Option 2: Search by text in payload (if stored)
        try:
            # First, let's try to get collection info
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
            
            # Get collection info to understand the structure
            collection_info = client.get_collection(collection_name)
            logger.info(f"Collection info: {collection_info}")
            
            # Try to search using scroll (get all points and filter by text similarity)
            # This is a basic approach - ideally you'd use vector search
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Filter results based on text similarity (basic keyword matching)
            relevant_chunks = []
            query_words = decoded_query.lower().split()
            
            for point in scroll_result[0]:  # scroll_result is a tuple (points, next_page_offset)
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
                        "content": matched_content,
                        "payload": payload
                    })
            
            # Sort by relevance score
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            relevant_chunks = relevant_chunks[:limit]
            
            return jsonify({
                "query": decoded_query,
                "collection": collection_name,
                "results": relevant_chunks,
                "total_found": len(relevant_chunks),
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
                    "status": info.status
                })
            except Exception as e:
                logger.warning(f"Could not get info for collection {collection.name}: {e}")
                collection_list.append({
                    "name": collection.name,
                    "points_count": "unknown",
                    "status": "unknown"
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
        
        # Try to get collections as a health check
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
        "message": "Qdrant Query Service",
        "endpoints": {
            "/query/<query_text>": "Search for relevant chunks",
            "/collections": "List available collections",
            "/health": "Health check"
        },
        "example": "GET /query/what%20is%20lingo?collection=your_collection&limit=5&threshold=0.1",
        "parameters": {
            "collection": "Collection name (optional)",
            "limit": "Maximum results to return (default: 10)",
            "threshold": "Minimum relevance score (default: 0.0)"
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