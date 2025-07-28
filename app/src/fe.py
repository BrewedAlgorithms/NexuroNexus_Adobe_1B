import pandas as pd
import numpy as np
import os
import re

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers a set of 30 features for document structure analysis.

    Args:
        df: A pandas DataFrame containing the raw data from the extract script.

    Returns:
        A pandas DataFrame with the added feature columns.
    """

    # Handle empty or nearly empty CSVs gracefully
    if df.shape[0] < 2:
        return df # Not enough data to create relational features

    # Ensure correct sorting for relational features
    df = df.sort_values(by=['page_number', 'bbox_y0']).reset_index(drop=True)

    # --- 1. Document-level Statistics ---
    doc_mode_font_size = df['font_size'].mode()[0] if not df['font_size'].empty else 10
    doc_avg_font_size = df['font_size'].mean()

    # --- 2. Positional & Size Features ---
    df['relative_y0'] = df['bbox_y0'] / df['page_height']
    df['relative_x0'] = df['bbox_x0'] / df['page_width']
    x_center = (df['bbox_x0'] + df['bbox_x1']) / 2
    df['horizontal_center_offset'] = np.abs((x_center / df['page_width']) - 0.5)
    df['relative_block_width'] = (df['bbox_x1'] - df['bbox_x0']) / df['page_width']
    df['relative_block_height'] = (df['bbox_y1'] - df['bbox_y0']) / df['page_height']
    df['relative_block_area'] = df['relative_block_width'] * df['relative_block_height']
    df['block_aspect_ratio'] = df['relative_block_width'] / df['relative_block_height']
    df['is_top_of_page'] = (df['relative_y0'] < 0.15).astype(int)
    df['is_bottom_of_page'] = (df['relative_y0'] > 0.85).astype(int)
    doc_left_margin = df['relative_x0'].quantile(0.1)
    df['is_left_aligned'] = (df['relative_x0'] < doc_left_margin + 0.05).astype(int)
    df['is_horizontally_centered'] = (df['horizontal_center_offset'] < 0.1).astype(int)

    # --- 3. Style & Content Features ---
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['text_length'] = df['text'].str.len()
    df['avg_word_length'] = df['text_length'] / df['word_count']
    df['is_all_caps'] = df['text'].apply(lambda x: str(x).isupper() and any(c.isalpha() for c in str(x))).astype(int)
    df['starts_with_numbering'] = df['text'].str.strip().str.match(r'^\d+[\.)\s]', na=False).astype(int)
    # ROBUSTNESS FIX: Added .fillna(False) to handle potentially empty text blocks (NaNs)
    df['ends_with_colon'] = df['text'].str.strip().str.endswith(':').fillna(False).astype(int)
    df['is_non_black_color'] = (df['font_color'] != '#000000').astype(int)
    df['char_density'] = df['text_length'] / (df['relative_block_area'] * 1000)
    df['digit_to_char_ratio'] = df['text'].str.count(r'[0-9]') / df['text_length']

    # --- 4. Contextual & Relational Features (requires sorted df) ---
    # Create shifted columns for previous/next block properties
    for col in ['page_number', 'bbox_y0', 'bbox_y1', 'bbox_x0', 'font_size', 'font_name', 'is_bold']:
        df[f'prev_{col}'] = df[col].shift(1)
        df[f'next_{col}'] = df[col].shift(-1)

    page_break = df['page_number'] != df['prev_page_number']
    next_page_break = df['page_number'] != df['next_page_number']

    df['vertical_space_before'] = df['bbox_y0'] - df['prev_bbox_y1']
    df.loc[page_break, 'vertical_space_before'] = np.nan

    df['vertical_space_after'] = df['next_bbox_y0'] - df['bbox_y1']
    df.loc[next_page_break, 'vertical_space_after'] = np.nan

    df['font_size_change_from_prev'] = df['font_size'] - df['prev_font_size']
    df.loc[page_break, 'font_size_change_from_prev'] = np.nan

    df['indentation_change_from_prev'] = df['bbox_x0'] - df['prev_bbox_x0']
    df.loc[page_break, 'indentation_change_from_prev'] = np.nan

    # ROBUSTNESS FIX: Added .fillna(False) to handle the first row where prev_* is NaN
    df['is_bold_change_from_prev'] = (df['is_bold'] != df['prev_is_bold']).fillna(False).astype(int)
    df.loc[page_break, 'is_bold_change_from_prev'] = np.nan

    # ROBUSTNESS FIX: Added .fillna(False) to handle the first row where prev_* is NaN
    df['is_font_change_from_prev'] = (df['font_name'] != df['prev_font_name']).fillna(False).astype(int)
    df.loc[page_break, 'is_font_change_from_prev'] = np.nan

    df['font_size_vs_doc_mode'] = df['font_size'] - doc_mode_font_size
    df['font_size_vs_doc_avg'] = df['font_size'] - doc_avg_font_size

    # --- 5. Cleanup ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop(columns=[col for col in df.columns if col.startswith('prev_') or col.startswith('next_')], inplace=True)

    return df

# --- New Main Function ---
def run_feature_engineering(input_dir, output_dir):
    """
    Adds features to all CSVs in input_dir and saves them in output_dir.
    """
    print("\n--- 2. Feature Engineering ---")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"🟡 No CSV files found in '{input_dir}'.")
        return
        
    print(f"Engineering features for {len(csv_files)} CSVs...")
    for filename in sorted(csv_files):
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"   -> Processing: {filename}")

            df = pd.read_csv(input_path)
            df_featured = create_features(df)
            df_featured.to_csv(output_path, index=False)

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    print("✅ Feature engineering complete.")