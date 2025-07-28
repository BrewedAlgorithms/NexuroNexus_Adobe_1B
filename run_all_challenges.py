# In run_all_challenges.py

import os
import sys
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).parent / 'app' / 'src'))

# Import the main pipeline function from the designated orchestrator module
from app.src.sub_sec_extract import execute_challenge

def find_challenge_collections(base_dir):
    """Finds all valid collection directories within the base directory."""
    collections = []
    # Look for directories like 'Collection 1', 'Collection 2', etc.
    for item in sorted(os.listdir(base_dir)):
        item_path = Path(base_dir) / item
        if (item_path.is_dir() and
            item.startswith("Collection") and # Make it specific
            (item_path / 'PDFs').is_dir() and
            (item_path / 'challenge1b_input.json').is_file()):
            collections.append(item_path)
    return collections

if __name__ == '__main__':
    base_directory = Path(__file__).parent
    print(f"🔍 Searching for collections in: {base_directory}")
    
    challenge_collections = find_challenge_collections(base_directory)
    
    if not challenge_collections:
        print("❌ No valid challenge collections found.")
        print("Ensure subdirectories are named 'Collection X' and contain 'PDFs/' and 'challenge1b_input.json'")
    else:
        print(f"✅ Found {len(challenge_collections)} collections to process.")
        for collection_path in challenge_collections:
            print("\n" + "="*80)
            print(f"🚀 Starting processing for: {collection_path.name}")
            print("="*80)
            
            # Define the specific input/output paths for this collection
            pdf_dir = collection_path / 'PDFs'
            input_json_path = collection_path / 'challenge1b_input.json'
            output_json_path = collection_path / 'challenge1b_output.json'
            
            try:
                # Execute the end-to-end pipeline for the current collection
                execute_challenge(
                    collection_path=collection_path,
                    pdf_dir=pdf_dir,
                    input_json_path=input_json_path,
                    output_json_path=output_json_path
                )
                print(f"\n✅ Successfully completed processing for {collection_path.name}")
                print(f"   -> Final output saved to: {output_json_path}")
            except Exception as e:
                print(f"\n❌ An error occurred while processing {collection_path.name}: {e}")
            
    print("\n\nAll collections processed.")