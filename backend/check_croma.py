import chromadb
import pprint

# --- Configuration ---
# According to your docker-compose.dev.yml, ChromaDB is running on port 8000.
# Your config.py file mentions port 8001, so you may need to adjust this
# if you have changed the default setup.
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000 

# --- Main Inspection Logic ---
def inspect_chroma():
    """Connects to ChromaDB and inspects its contents."""
    
    print(f"Attempting to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    
    try:
        # Use the HttpClient to connect to the running ChromaDB service
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        # Check if the connection is alive
        client.heartbeat() 
        print(f"✅ Successfully connected to ChromaDB version")
        
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB.")
        print(f"   Error: {e}")
        print("\n   Please ensure the ChromaDB Docker container is running.")
        print("   You can start it with: docker-compose -f docker-compose.dev.yml up -d chromadb")
        return

    # 1. List all collections
    collections = client.list_collections()
    if not collections:
        print("\nNo collections found in ChromaDB.")
        return
        
    print(f"\nFound {len(collections)} collections:")
    for collection in collections:
        print(f"- {collection.name} (ID: {collection.id})")

    # 2. Inspect the first collection found
    # You can change this to a specific collection name if you know it.
    collection_to_inspect_name = collections[0].name
    print(f"\nInspecting contents of collection: '{collection_to_inspect_name}'")
    
    try:
        collection = client.get_collection(name=collection_to_inspect_name)
        
        # Get the total number of items
        count = collection.count()
        print(f"Total items (vectors) in collection: {count}")
        
        if count > 0:
            # Retrieve a few items to inspect their structure
            # This fetches the document content, metadata, and embeddings.
            items = collection.get(
                limit=5,
                include=["metadatas", "documents"] # You can also include "embeddings"
            )
            
            print("\nSample of items stored in the collection:")
            pprint.pprint(items)
            
    except Exception as e:
        print(f"Error inspecting collection '{collection_to_inspect_name}': {e}")


if __name__ == "__main__":
    inspect_chroma()
