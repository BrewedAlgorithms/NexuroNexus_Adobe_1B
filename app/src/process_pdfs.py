# In app/src/process_pdfs.py

import os
from pathlib import Path

# Import the main functions from the other src modules
from .extract import run_extraction
from .fe import run_feature_engineering
from .ml_model import run_prediction
from .apply_heuristics import run_heuristics
from .json_output_2 import run_json_conversion

def process_pdfs(pdf_input_dir: Path, collection_processing_dir: Path, final_json_output_dir: Path):
    """
    Orchestrates the PDF-to-structured-JSON pipeline for a single collection.
    
    Args:
        pdf_input_dir: The directory with the source PDFs.
        collection_processing_dir: The base directory for all intermediate files.
        final_json_output_dir: The directory where the final structured JSONs will be saved.
    """
    print("--- [Step 2.1: Running PDF-to-JSON Conversion] ---")

    # --- 1. Define and create all necessary directories FOR THIS COLLECTION ---
    EXTRACTED_CSV_DIR = collection_processing_dir / "01_extracted"
    FEATURED_CSV_DIR = collection_processing_dir / "02_featured"
    PREDICTED_CSV_DIR = collection_processing_dir / "03_predicted"
    CORRECTED_CSV_DIR = collection_processing_dir / "04_corrected"

    # The final_json_output_dir is also managed here
    for path in [EXTRACTED_CSV_DIR, FEATURED_CSV_DIR, PREDICTED_CSV_DIR, CORRECTED_CSV_DIR, final_json_output_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if not any(pdf_input_dir.glob("*.pdf")):
        print(f"🟡 No PDF files found in '{pdf_input_dir}'. Skipping PDF processing.")
        return

    # --- 2. Execute Pipeline Steps Sequentially ---
    # The 'model' directory is assumed to be in the 'app' folder
    model_dir = Path(__file__).parent.parent / "model"

    run_extraction(pdf_dir=pdf_input_dir, output_dir=EXTRACTED_CSV_DIR)
    run_feature_engineering(input_dir=EXTRACTED_CSV_DIR, output_dir=FEATURED_CSV_DIR)
    run_prediction(test_dir=FEATURED_CSV_DIR, model_dir=model_dir, output_dir=PREDICTED_CSV_DIR)
    run_heuristics(input_dir=PREDICTED_CSV_DIR, output_dir=CORRECTED_CSV_DIR)
    # This step produces the final structured JSONs needed for semantic search
    run_json_conversion(input_dir=CORRECTED_CSV_DIR, output_dir=final_json_output_dir)

    print(f"✅ PDF-to-JSON conversion complete. Structured JSONs are in '{final_json_output_dir}'.")