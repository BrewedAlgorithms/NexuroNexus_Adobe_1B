import json
import os
import torch
import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Import the main function from the PDF processing pipeline
from .process_pdfs import process_pdfs
# Import the main function from the new section extraction script
from .sec_extract import extract_top_sections

# ==============================================================================
# STEP 1: LOAD SUBSECTION ANALYSIS MODELS
# ==============================================================================
print("Loading models for subsection analysis...")
# Only the bi-encoder is needed for sentence ranking
try:
    bi_encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Subsection analysis model loaded successfully.")
except Exception as e:
    print(f"Could not load bi-encoder model for subsection analysis. Error: {e}")
    bi_encoder_model = None

# ==============================================================================
# PART 1: SUBSECTION ANALYSIS (Refactored)
# ==============================================================================
def analyze_subsections(ranked_sections: list, query: str, top_n_sections=5, top_m_sentences=5):
    """
    Extracts and ranks individual sentences from the pre-ranked sections.
    """
    print("\n--- [Step 4: Performing Subsection Analysis] ---")
    if not bi_encoder_model or not ranked_sections:
        print("Subsection model not loaded or no ranked sections provided. Skipping.")
        return []

    sentences_to_rank = []
    
    # 1. Collect sentences from the top N sections
    for section in ranked_sections[:top_n_sections]:
        content = section.get('content', '')
        # Simple regex to split sentences, handles more punctuation
        sentences = re.split(r'(?<=[.?!])\s+', content)
        for sentence in sentences:
            cleaned_sentence = sentence.strip()
            # Consider sentences with a reasonable length
            if 5 < len(cleaned_sentence.split()) < 100:
                sentences_to_rank.append({
                    "text": cleaned_sentence,
                    "document": section.get('document'),
                    "page_number": section.get('page_number')
                })

    if not sentences_to_rank:
        print("No suitable sentences found for subsection analysis.")
        return []

    # 2. Rank sentences using the bi-encoder against the rich query
    query_emb = bi_encoder_model.encode(query, convert_to_tensor=True)
    sentence_texts = [s['text'] for s in sentences_to_rank]
    sentence_embs = bi_encoder_model.encode(sentence_texts, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_emb, sentence_embs)[0]
    
    # 3. Combine sentences with their scores and sort
    ranked_sentences = sorted(zip(sentences_to_rank, cos_scores.tolist()), key=lambda x: x[1], reverse=True)
    
    # 4. Format the top M sentences for the final output
    subsection_results = []
    for sentence_info, score in ranked_sentences[:top_m_sentences]:
        subsection_results.append({
            "document": sentence_info['document'],
            "refined_text": sentence_info['text'],
            "page_number": sentence_info['page_number'],
            "relevance_score": round(score, 4)
        })
        
    print(f"Extracted {len(subsection_results)} relevant subsections.")
    return subsection_results


# ==============================================================================
# PART 2: MAIN PIPELINE ORCHESTRATOR
# ==============================================================================
def execute_challenge(collection_path: Path, pdf_dir: Path, input_json_path: Path, output_json_path: Path):
    """
    Orchestrates the entire pipeline for a single challenge collection.
    """
    # --- 1. Setup processing directories for this specific collection ---
    PROCESSING_DIR = collection_path / "temp_processing"
    STRUCTURED_JSON_DIR = PROCESSING_DIR / "structured_json_output"
    RANKED_SECTIONS_PATH = PROCESSING_DIR / "ranked_sections.json"
    
    if PROCESSING_DIR.exists():
        shutil.rmtree(PROCESSING_DIR)
    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"--- [Step 1: Reading Challenge Input] ---")
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)

    # --- 2. Run PDF to Structured JSON Pipeline ---
    process_pdfs(
        pdf_input_dir=pdf_dir, 
        collection_processing_dir=PROCESSING_DIR, 
        final_json_output_dir=STRUCTURED_JSON_DIR
    )

    # --- 3. Run Section Extraction and Ranking ---
    my_role = input_data['persona']['role']
    my_task = input_data['job_to_be_done']['task']
    
    # This function now handles query generation and section ranking, saving the result
    rich_query = extract_top_sections(
        structured_json_dir=STRUCTURED_JSON_DIR,
        role=my_role,
        task=my_task,
        output_path=RANKED_SECTIONS_PATH
    )

    # --- 4. Perform Subsection Analysis ---
    with open(RANKED_SECTIONS_PATH, 'r', encoding='utf-8') as f:
        top_sections = json.load(f)

    subsection_analysis = analyze_subsections(top_sections, rich_query)

    # --- 5. Assemble Final Output ---
    print("\n--- [Step 5: Assembling Final Output] ---")
    extracted_sections_output = []
    # Format the top 5 sections for the final output
    for i, section in enumerate(top_sections[:5], 1):
        extracted_sections_output.append({
            "document": section.get('document'),
            "section_title": section.get('header'),
            "importance_rank": i,
            "page_number": section.get('page_number')
        })

    metadata = {
        "input_documents": [doc['filename'] for doc in input_data['documents']],
        "persona": my_role,
        "job_to_be_done": my_task,
        "generated_query": rich_query,
        "processing_timestamp": datetime.utcnow().isoformat()
    }
    
    final_output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_analysis
    }

    # --- 6. Write Final JSON and Clean Up ---
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n✅✅✅ Challenge complete! Final output saved to '{output_json_path}'")
    
    # Optional: clean up temporary files
    # shutil.rmtree(PROCESSING_DIR)