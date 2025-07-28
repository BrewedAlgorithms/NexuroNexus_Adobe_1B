import pandas as pd
import os


def apply_rules_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a set of heuristic rules to correct the predicted labels in a DataFrame.
    """
    # Ensure data is sorted in reading order
    df = df.sort_values(by=['page_number', 'bbox_y0']).reset_index(drop=True)

    # --- Rule 1: Title can only be on pages 1-3 and there can be only one. ---
    # Find all rows predicted as 'title'
    title_candidates = df[df['predicted_label'] == 'title'].copy()
    
    # Filter to pages 1, 2, or 3
    title_candidates = title_candidates[title_candidates['page_number'].isin([1, 2, 3])]
    
    true_title_index = None
    if not title_candidates.empty:
        # Select the 'best' title (e.g., largest font size, then earliest appearance)
        title_candidates = title_candidates.sort_values(by=['font_size', 'page_number', 'bbox_y0'], ascending=[False, True, True])
        true_title_index = title_candidates.index[0]
        
        # Get the indices of all other 'title' predictions that are now invalid
        invalid_title_indices = df[(df['predicted_label'] == 'title') & (df.index != true_title_index)].index
        
        # Demote invalid titles to 'other'
        if not invalid_title_indices.empty:
            print(f"   -> Rule 1: Demoting {len(invalid_title_indices)} invalid 'title' predictions.")
            df.loc[invalid_title_indices, 'predicted_label'] = 'other'

    # --- Rule 2: No h1s before the title. ---
    if true_title_index is not None:
        # Find all h1s that appear before the true title
        invalid_h1s = df[(df['predicted_label'] == 'h1') & (df.index < true_title_index)]
        
        if not invalid_h1s.empty:
            print(f"   -> Rule 2: Demoting {len(invalid_h1s)} 'h1' predictions that appeared before the title.")
            df.loc[invalid_h1s.index, 'predicted_label'] = 'other'
            
    # --- Rule 3: Enforce heading hierarchy (no h2 without h1, no h3 without h2). ---
    last_seen_heading_level = 0
    demoted_count_hierarchy = 0
    for index, row in df.iterrows():
        label = row['predicted_label']
        
        if label == 'title':
            last_seen_heading_level = 0 # Reset hierarchy after title
        elif label == 'h1':
            last_seen_heading_level = 1
        elif label == 'h2':
            if last_seen_heading_level < 1:
                df.loc[index, 'predicted_label'] = 'other' # Demote h2
                demoted_count_hierarchy += 1
            else:
                last_seen_heading_level = 2
        elif label == 'h3':
            if last_seen_heading_level < 2:
                df.loc[index, 'predicted_label'] = 'other' # Demote h3
                demoted_count_hierarchy += 1
            else:
                last_seen_heading_level = 3
    
    if demoted_count_hierarchy > 0:
        print(f"   -> Rule 3: Demoted {demoted_count_hierarchy} headings that violated structural hierarchy.")

    # --- Rule 4: Enforce font size hierarchy (title >= h1 >= h2 >= h3). ---
    title_font_size = float('inf')
    if true_title_index is not None:
        # Check if the title still exists after previous rules
        if df.loc[true_title_index, 'predicted_label'] == 'title':
             title_font_size = df.loc[true_title_index, 'font_size']
        else:
            # The original title was demoted by another rule.
            true_title_index = None


    last_h1_font_size = float('inf')
    last_h2_font_size = float('inf')
    demoted_count_fontsize = 0
    
    for index, row in df.iterrows():
        label = row['predicted_label']
        font_size = row['font_size']

        if label == 'title':
            # This should only be the one true title, reset font hierarchy
            last_h1_font_size = float('inf')
            last_h2_font_size = float('inf')
        elif label == 'h1':
            if font_size > title_font_size:
                df.loc[index, 'predicted_label'] = 'other'
                demoted_count_fontsize += 1
            else:
                last_h1_font_size = font_size
                last_h2_font_size = float('inf') # Reset h2 font size on new h1
        elif label == 'h2':
            if font_size > last_h1_font_size:
                df.loc[index, 'predicted_label'] = 'other'
                demoted_count_fontsize += 1
            else:
                last_h2_font_size = font_size
        elif label == 'h3':
            if font_size > last_h2_font_size:
                df.loc[index, 'predicted_label'] = 'other'
                demoted_count_fontsize += 1

    if demoted_count_fontsize > 0:
        print(f"   -> Rule 4: Demoted {demoted_count_fontsize} headings that violated font size hierarchy.")

    return df

def run_heuristics(input_dir, output_dir):
    """
    Applies correction rules to predicted CSVs from input_dir and saves them to output_dir.
    """
    print("\n--- 4. Applying Heuristics ---")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"🟡 No CSV files found in '{input_dir}' to process.")
        return

    print(f"Applying heuristic rules to {len(csv_files)} files...")
    for filename in sorted(csv_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print(f"\n   -> Correcting: {filename}")
        
        df = pd.read_csv(input_path)
        df_corrected = apply_rules_to_dataframe(df)
        
        df_corrected.to_csv(output_path, index=False)
        print(f"      -> Saved final predictions to '{os.path.basename(output_path)}'")
    
    print(f"\n✅ Heuristics application complete.")