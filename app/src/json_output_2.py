import pandas as pd
import json
import os
import argparse

def create_structured_json(input_csv_path, output_json_path):
    """
    Reads a corrected CSV file and builds a JSON structure based on h1 headings.

    The output format is a list of objects, where each object contains:
    - "header": The text of an 'h1' labeled element.
    - "page_number": The page number where the 'h1' header is located.
    - "content": A single string containing all text from subsequent elements
                 until the next 'h1' or the end of the document.
    """
    try:
        df = pd.read_csv(input_csv_path)
        if df.empty:
            print(f"   -> Skipping empty CSV: {os.path.basename(input_csv_path)}")
            # Write an empty list to the JSON file for empty inputs
            with open(output_json_path, 'w', encoding='utf-8') as outfile:
                json.dump([], outfile)
            return

        # Sort by reading order to ensure content is sequential
        df = df.sort_values(by=['page_number', 'bbox_y0']).reset_index(drop=True)

        # Find the indices of all rows labeled as 'h1'
        h1_indices = df[df['predicted_label'] == 'h1'].index.tolist()

        # If no h1 headings are found, produce an empty JSON list
        if not h1_indices:
            print(f"   -> No 'h1' headings found in {os.path.basename(input_csv_path)}. Output will be empty.")
            with open(output_json_path, 'w', encoding='utf-8') as outfile:
                json.dump([], outfile, indent=2)
            return

        json_output_list = []

        # Iterate through each h1 to define content sections
        for i, start_index in enumerate(h1_indices):
            # The start of content is the row immediately after the h1 header
            start_content_index = start_index + 1
            
            # The end of content is the index of the next h1, or the end of the dataframe
            if i + 1 < len(h1_indices):
                end_content_index = h1_indices[i + 1]
            else:
                end_content_index = len(df)
            
            # Extract header text and page number from the h1 row
            header_row = df.loc[start_index]
            header_text = header_row['text']
            page_number = int(header_row['page_number']-1) # Get the page number

            # Slice the dataframe to get all content rows for the current section
            content_df = df.iloc[start_content_index:end_content_index]

            # Join the 'text' of all content rows into a single string
            content_text = ' '.join(content_df['text'].astype(str).tolist())
            
            json_output_list.append({
                "header": header_text,
                "page_number": page_number,
                "content": content_text
            })

        # Write the structured list to the final JSON file
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(json_output_list, outfile, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"❌ An error occurred while creating JSON for {os.path.basename(input_csv_path)}: {e}")

def run_json_conversion(input_dir, output_dir):
    """
    Main function to convert all CSVs in a directory to the structured JSON format.
    """
    print(f"\n--- Structured JSON Conversion (h1 sections) ---")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"🟡 No CSV files found in '{input_dir}' to convert.")
        return

    print(f"Converting {len(csv_files)} CSVs from '{input_dir}' to JSON in '{output_dir}'...")
    for filename in sorted(csv_files):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.json"
        output_path = os.path.join(output_dir, output_filename)
        print(f"   -> Generating {output_filename}")
        create_structured_json(input_path, output_path)
    print(f"✅ JSON conversion complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert final CSVs to structured JSON based on h1 sections.")
    parser.add_argument('--input_dir', type=str, default='output', help='Directory with corrected CSV files.')
    parser.add_argument('--output_dir', type=str, default='json_output_h1', help='Directory to save final structured JSON files.')
    args = parser.parse_args()
    run_json_conversion(args.input_dir, args.output_dir)