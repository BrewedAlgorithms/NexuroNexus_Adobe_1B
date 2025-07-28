import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
import argparse
from pathlib import Path

# ==============================================================================
# STEP 1: LOAD ALL MODELS
# ==============================================================================
print("Loading all AI models for section extraction...")

# Model for Rich Query Generation
try:
    query_gen_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    query_gen_tokenizer = AutoTokenizer.from_pretrained(query_gen_model_id)
    query_gen_model = AutoModelForCausalLM.from_pretrained(
        query_gen_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
except Exception as e:
    print(f"Could not load TinyLlama model. Error: {e}")
    query_gen_model = None

# Models for Ranking
try:
    bi_encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    device = 0 if torch.cuda.is_available() else -1
    nli_model = pipeline("zero-shot-classification",
                         model="typeform/distilbert-base-uncased-mnli",
                         device=device)
    print("All ranking models loaded successfully.")
except Exception as e:
    print(f"Could not load ranking models. Error: {e}")
    bi_encoder_model = None
    nli_model = None

# ==============================================================================
# STEP 2: DEFINE CORE FUNCTIONS
# ==============================================================================

def generate_rich_query(role: str, task: str) -> str:
    """Uses TinyLlama to generate a rich query from a high-level goal."""
    if not query_gen_model:
        print("Query generation model not loaded. Using a basic query.")
        return f"As a {role}, I need to find information to help me {task}."

    messages = [
        {"role": "system", "content": "You are an expert at creating rich, descriptive search statements. Synthesize the user's goal into a single, detailed paragraph to be used as a search query. Do NOT use bullet points or lists."},
        {"role": "user", "content": f"As a {role}, I need to find information to help me {task}."},
    ]
    prompt = query_gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = query_gen_tokenizer(prompt, return_tensors="pt").to(query_gen_model.device)
    outputs = query_gen_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.3, top_p=0.95)
    text = query_gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = text.split("<|assistant|>")[-1].strip()
    return response


def load_and_prepare_sections(folder_path: Path) -> List[Dict[str, Any]]:
    """Scans a directory for .json files, loads all sections, and adds source info."""
    all_sections = []
    if not folder_path.is_dir():
        print(f"Error: Directory not found at '{folder_path}'")
        return all_sections

    print(f"\nScanning for JSON files in '{folder_path}'...")
    for file_path in folder_path.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for section in data:
                        section['document'] = file_path.with_suffix(".pdf").name
                        all_sections.append(section)
                    print(f"  - Loaded {len(data)} sections from {file_path.name}")
        except Exception as e:
            print(f"  - ERROR loading {file_path.name}: {e}")
    return all_sections


def rank_sections_with_query(query: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ranks sections using a two-stage process: Bi-encoder then NLI model."""
    if not bi_encoder_model or not nli_model or not sections:
        print("Ranking models not loaded or no sections to rank.")
        return []

    # Stage 1: Bi-encoder for initial candidate search
    query_emb = bi_encoder_model.encode(query, convert_to_tensor=True)
    # Use .get() for safety in case 'header' or 'content' is missing
    sec_texts = [f"{s.get('header', '')}. {s.get('content', '')}" for s in sections]
    sec_embs = bi_encoder_model.encode(sec_texts, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_emb, sec_embs)[0]
    top_k = min(len(sections), 20)

    topk_indices = cos_scores.topk(k=top_k).indices.tolist()
    candidates = [sections[i] for i in topk_indices]
    candidate_texts = [sec_texts[i] for i in topk_indices]

    # Stage 2: NLI model for accurate re-ranking
    hypothesis = f"This section is relevant for the following task: {query}"
    nli_results = nli_model(candidate_texts, candidate_labels=[hypothesis], multi_label=False)

    # Extract the 'entailment' score which is usually the first one
    entail_scores = [res['scores'][0] for res in nli_results]
    
    # Pair candidates with their scores and sort
    ranked_pairs = sorted(zip(candidates, entail_scores), key=lambda x: x[1], reverse=True)

    # Filter out low-quality results and format final list
    final_ranked_list = []
    for section, score in ranked_pairs:
        # Filter out sections with common non-informative titles
        if section.get('header', '').strip().lower() not in ['conclusion', 'introduction', 'references']:
            section['relevance_score'] = score
            final_ranked_list.append(section)
            
    return final_ranked_list

# ==============================================================================
# STEP 3: MAIN EXECUTION FUNCTION
# ==============================================================================
def extract_top_sections(structured_json_dir: Path, role: str, task: str, output_path: Path):
    """
    Main function to orchestrate the section extraction and ranking process.
    """
    print("\n--- [Running High-Level Section Extraction] ---")

    # 1. Load and Prepare Data
    print("\n--- [Loading and Preparing Data] ---")
    sections_for_ranking = load_and_prepare_sections(structured_json_dir)
    if not sections_for_ranking:
        print("Error: No sections found. Cannot proceed.")
        # Write an empty list to the output to prevent downstream errors
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return

    # 2. Generate Rich Query
    print("\n--- [Generating Rich Query] ---")
    rich_query = generate_rich_query(role, task)
    print(f"Generated Query: {rich_query}")

    # 3. Rank Sections
    print("\n--- [Ranking Sections with Two-Stage Model] ---")
    final_ranking = rank_sections_with_query(rich_query, sections_for_ranking)
    print(f"Ranked {len(final_ranking)} sections.")

    # 4. Save Ranked Sections to JSON file
    print(f"\n--- [Saving Ranked Sections to '{output_path.name}'] ---")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_ranking, f, indent=4)
    
    print("✅ Section extraction and ranking complete.")
    return rich_query # Return the query for the next step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract and rank relevant sections from structured JSON files.")
    parser.add_argument('--input_dir', type=Path, required=True, help='Directory containing the structured JSON files from the PDF pipeline.')
    parser.add_argument('--output_path', type=Path, required=True, help='File path to save the ranked sections JSON.')
    parser.add_argument('--role', type=str, required=True, help='The user role or persona.')
    parser.add_argument('--task', type=str, required=True, help='The job to be done or task.')
    args = parser.parse_args()

    extract_top_sections(
        structured_json_dir=args.input_dir,
        role=args.role,
        task=args.task,
        output_path=args.output_path
    )