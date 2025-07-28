import fitz  # PyMuPDF
import os
import csv
import statistics
import re

# --- Configuration ---
PDF_DIR = "pdfs"
EXTRACT_DIR = "extract"

# --- Setup ---
os.makedirs(EXTRACT_DIR, exist_ok=True)

def clean_text(text):
    """Removes unwanted newlines and extra spaces from the text."""
    # First, replace sequences of whitespace characters (including newlines) with a single space.
    # Then, strip leading/trailing whitespace.
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def flags_to_style(flags):
    """Converts PyMuPDF font flags to a style dictionary."""
    style = {
        'is_bold': bool(flags & 2**4),
        'is_italic': bool(flags & 2**1)
    }
    return style

def color_int_to_hex(color_int):
    """Converts an integer color value to a hex string."""
    if color_int is None:
        return "#000000" # Default to black
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"

def split_embedded_heading_chunk(chunk):
    """
    Splits a chunk if it contains an embedded heading based on specific rules.
    Example: "The Boy: The most significant thing in his life was a..."
    Returns a list containing one or two chunks.
    """
    text = chunk['text']

    # Rule: Must contain a colon to be a candidate for splitting.
    if ':' not in text:
        return [chunk]

    parts = text.split(':', 1)
    left_part = parts[0].strip()
    right_part = parts[1].strip()

    # Rule: Left part (potential heading) must be between 1 and 8 words. (MODIFIED)
    left_word_count = len(left_part.split())
    if not (1 <= left_word_count <= 8):
        return [chunk]

    # Rule: Right part (content) must be more than 10 words long.
    right_word_count = len(right_part.split())
    if right_word_count <= 10:
        return [chunk]

    # Rule: Right part must not contain any other colons.
    if ':' in right_part:
        return [chunk]

    # --- If all rules pass, proceed with splitting the chunk ---

    # Estimate the bounding box split point based on character length
    total_len = len(text)
    # Include the colon in the left part's length for bbox calculation
    left_len_for_bbox = len(left_part) + 1
    split_ratio = left_len_for_bbox / total_len if total_len > 0 else 0

    original_bbox = chunk['bbox']
    # Calculate the horizontal split point
    split_x = original_bbox.x0 + (original_bbox.width * split_ratio)

    # Create the left chunk (the "embedded heading")
    left_chunk = chunk.copy()
    left_chunk['text'] = left_part + ":" # Keep the colon with the heading
    left_chunk['bbox'] = fitz.Rect(original_bbox.x0, original_bbox.y0, split_x, original_bbox.y1)
    left_chunk['is_bold'] = True # MODIFIED: Force embedded heading to be bold

    # Create the right chunk (the content)
    right_chunk = chunk.copy()
    right_chunk['text'] = right_part
    right_chunk['bbox'] = fitz.Rect(split_x, original_bbox.y0, original_bbox.x1, original_bbox.y1)

    return [left_chunk, right_chunk]


def analyze_and_extract_content(pdf_path):
    """
    Analyzes a PDF to identify paragraph/heading styles and extracts detailed content,
    ensuring that text chunks with different styles are not merged.
    
    Returns:
        A list of dictionaries, where each dictionary represents a content chunk.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"   -> Failed to open {os.path.basename(pdf_path)}: {e}")
        return []
        
    # --- Pass 1: Collect font sizes from all individual spans for accurate analysis ---
    font_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0: # Text block
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        if span['text'].strip():
                            font_sizes.append(round(span['size']))

    if not font_sizes:
        print(f"   -> No text content found in {os.path.basename(pdf_path)}")
        doc.close()
        return []

    # Determine the most common (paragraph) font size
    try:
        para_size = statistics.mode(font_sizes)
    except statistics.StatisticsError:
        # Fallback if no single mode exists (e.g., all sizes are unique)
        para_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 12

    # --- Pass 2: Extract all spans and their styles individually ---
    raw_chunks = []
    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        page_height = page.rect.height
        
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0: # Text block
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        # Skip empty spans
                        if not span['text'].strip():
                            continue
                            
                        style_info = flags_to_style(span['flags'])
                        raw_chunks.append({
                            'page_num': page_num + 1,
                            'page_width': page_width,
                            'page_height': page_height,
                            'font_size': round(span['size']),
                            'font_name': span['font'],
                            'font_color': color_int_to_hex(span['color']),
                            'is_bold': style_info['is_bold'],
                            'is_italic': style_info['is_italic'],
                            'bbox': fitz.Rect(span['bbox']), # Use fitz.Rect for easier manipulation
                            'text': span['text']
                        })

    if not raw_chunks:
        doc.close()
        return []

    # --- Pass 3: Merge consecutive spans that share the same style ---
    merged_chunks = []
    if raw_chunks:
        # Start with the first raw chunk
        current_chunk = raw_chunks[0]

        for next_chunk in raw_chunks[1:]:
            # Define style-matching criteria
            is_same_style = (
                current_chunk['font_size'] == next_chunk['font_size'] and
                current_chunk['font_name'] == next_chunk['font_name'] and
                current_chunk['font_color'] == next_chunk['font_color'] and
                current_chunk['is_bold'] == next_chunk['is_bold'] and
                current_chunk['is_italic'] == next_chunk['is_italic'] and
                current_chunk['page_num'] == next_chunk['page_num']
            )

            if is_same_style:
                # Merge: append text and union the bounding box
                current_chunk['text'] += " " + next_chunk['text']
                current_chunk['bbox'] |= next_chunk['bbox'] # Union of fitz.Rect objects
            else:
                # Styles are different, save the completed chunk and start a new one
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # Append the very last processed chunk
        merged_chunks.append(current_chunk)

    # --- Pass 4: Apply embedded heading rule, then classify and format the final output ---
    
    # Apply the new de-chunking logic before final classification
    dechunked_chunks = []
    for chunk in merged_chunks:
        # The new function returns a list, so extend the list with its contents
        dechunked_chunks.extend(split_embedded_heading_chunk(chunk))
    
    content_chunks = []
    for chunk in dechunked_chunks: # Iterate over the potentially de-chunked list
        # Clean the merged text at the very end
        cleaned_text = clean_text(chunk['text'])
        if not cleaned_text:
            continue

        # Classify the chunk type based on its font size relative to the paragraph size
        chunk_type = 'paragraph'
        if chunk['font_size'] > para_size:
            chunk_type = 'heading'
        elif chunk['font_size'] < para_size:
            chunk_type = 'footnote'

        bbox = chunk['bbox']
        
        # Append the final, formatted data, ensuring the output format is not changed
        content_chunks.append({
            'page_number': chunk['page_num'],
            'page_width': chunk['page_width'],
            'page_height': chunk['page_height'],
            'type': chunk_type,
            'text': cleaned_text,
            'font_size': chunk['font_size'],
            'font_name': chunk['font_name'],
            'font_color': chunk['font_color'],
            'is_bold': chunk['is_bold'],
            'is_italic': chunk['is_italic'],
            'bbox_x0': bbox.x0,
            'bbox_y0': bbox.y0,
            'bbox_x1': bbox.x1,
            'bbox_y1': bbox.y1,
        })
    
    doc.close()
    return content_chunks


# --- New Main Function ---
def run_extraction(pdf_dir, output_dir):
    """
    Extracts content from all PDFs in pdf_dir and saves them as CSVs in output_dir.
    """
    print("\n--- 1. PDF Content Extraction ---")
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"🟡 No PDFs found in '{pdf_dir}'.")
        return

    print(f"Extracting content from {len(pdf_files)} PDFs...")
    for pdf_filename in sorted(pdf_files):
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        print(f"   -> Processing: {pdf_filename}")

        extracted_data = analyze_and_extract_content(pdf_path)

        if not extracted_data:
            print(f"      -> No data extracted from {pdf_filename}.")
            continue

        csv_filename = os.path.splitext(pdf_filename)[0] + ".csv"
        csv_path = os.path.join(output_dir, csv_filename)

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                fieldnames = [
                    'page_number', 'page_width', 'page_height', 'type', 'text', 
                    'font_size', 'font_name', 'font_color', 'is_bold', 'is_italic',
                    'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1'
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(extracted_data)
        except IOError as e:
            print(f"      -> Error writing CSV: {e}")
    print("✅ Extraction complete.")