import chromadb
import pprint
import warnings

# --- Configuration ---
# According to your docker-compose.dev.yml, ChromaDB is running on port 8000.
# Your config.py file should also be set to 8000.
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# --- Main Inspection Logic ---
def inspect_chroma():
    """Connects to ChromaDB and inspects its contents."""
    
    print(f"Attempting to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    
    client = None
    try:
        # Suppress the specific telemetry warning if it occurs
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="capture() takes 1 positional argument but 3 were given")
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        # Check if the connection is alive
        version = client.version()
        print(f"✅ Successfully connected to ChromaDB version: {version}")
        
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB.")
        print(f"   Error: {e}")
        print("\n   Please ensure the ChromaDB Docker container is running.")
        print("   You can start it with: docker-compose -f docker-compose.dev.yml up -d chromadb")
        return

    try:
        # 1. List all collections
        collections = client.list_collections()
        if not collections:
            print("\n-> No collections found in ChromaDB.")
            print("   This means no documents have been successfully processed and indexed yet.")
            return
            
        print(f"\nFound {len(collections)} collections:")
        for collection in collections:
            print(f"- Name: {collection.name} (ID: {collection.id})")
            # You can also inspect metadata if needed:
            # print(f"  Metadata: {collection.metadata}")

        # 2. Inspect the first collection found
        # You can change this to a specific collection name if you know it.
        collection_to_inspect_name = collections[0].name
        print(f"\nInspecting contents of collection: '{collection_to_inspect_name}'")
        
        collection = client.get_collection(name=collection_to_inspect_name)
        
        # Get the total number of items
        count = collection.count()
        print(f"-> Total items (vectors) in collection: {count}")
        
        if count > 0:
            # Retrieve a few items to inspect their structure
            # This fetches the document content, metadata, and embeddings.
            print("\n-> Sample of items stored in the collection:")
            items = collection.get(
                limit=min(5, count),
                include=["metadatas", "documents"] # You can also include "embeddings"
            )
            pprint.pprint(items)
            
    except Exception as e:
        print(f"\n❌ Error inspecting collections: {e}")


if __name__ == "__main__":
    inspect_chroma()
