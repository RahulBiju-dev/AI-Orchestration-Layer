import time
import os
from unittest.mock import MagicMock
from tools.vault_indexer import delete_vault_item

def test_baseline():
    mock_collection = MagicMock()
    # Simulate existing data for the filters
    mock_collection.get.return_value = {"ids": [f"id{i}" for i in range(100)]}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    import tools.vault_indexer
    tools.vault_indexer.get_chroma_client = lambda: mock_client

    start_time = time.perf_counter()
    for _ in range(1000):
        delete_vault_item(source="test_source", collection_name="test_collection")
    end_time = time.perf_counter()

    print(f"Baseline Time: {end_time - start_time:.4f} seconds")
    print(f"Calls to get: {mock_collection.get.call_count}")
    print(f"Calls to delete: {mock_collection.delete.call_count}")

if __name__ == "__main__":
    test_baseline()
