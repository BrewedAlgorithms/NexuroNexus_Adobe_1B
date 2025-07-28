import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

# Define the features to be used for training.
# These should be the features created by fe.py
FEATURES = [
    'relative_y0', 'horizontal_center_offset', 'relative_block_width',
    'relative_block_height', 'relative_block_area', 'block_aspect_ratio',
    'is_top_of_page', 'is_bottom_of_page', 'is_left_aligned',
    'is_horizontally_centered', 'word_count', 'text_length',
    'avg_word_length', 'is_all_caps', 'starts_with_numbering',
    'ends_with_colon', 'is_non_black_color', 'char_density',
    'digit_to_char_ratio', 'vertical_space_before', 'vertical_space_after',
    'font_size_change_from_prev', 'indentation_change_from_prev',
    'is_bold_change_from_prev', 'is_font_change_from_prev',
    'font_size_vs_doc_mode', 'font_size_vs_doc_avg',
    # Original features that are still very important
    'font_size', 'is_bold', 'is_italic'
]

# Categorical features for LGBM
CATEGORICAL_FEATURES = ['font_name']


def load_data(directory: str, is_training: bool = True) -> pd.DataFrame:
    """Loads and concatenates all CSV files from a directory."""
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in the '{directory}' directory.")
    
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    if is_training:
        if 'label' not in df.columns:
            raise ValueError("The training CSVs must have a 'label' column.")
        df['label'] = df['label'].fillna('other')
        
    return df

def train_model():
    """Trains the LGBM model and saves it to disk."""
    print("--- Starting Model Training ---")
    
    print(f"Loading training data from '{TRAIN_DIR}'...")
    df_train = load_data(TRAIN_DIR, is_training=True)
    
    print("Preparing data for training...")
    df_train[FEATURES] = df_train[FEATURES].fillna(0)
    
    le = LabelEncoder()
    df_train['label_encoded'] = le.fit_transform(df_train['label'])
    
    if 'font_name' in df_train.columns:
        df_train['font_name'] = df_train['font_name'].astype('category')

    X = df_train[FEATURES + CATEGORICAL_FEATURES]
    y = df_train['label_encoded']

    print("Training LightGBM model...")
    lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=42)
    
    # Stratify helps ensure validation set represents class distribution, but can't fix very rare classes
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='multi_logloss', 
             callbacks=[lgb.early_stopping(10, verbose=False)], categorical_feature=CATEGORICAL_FEATURES)
    
    y_pred = lgbm.predict(X_val)
    print("\n--- Classification Report (on validation set) ---")
    
    # *** THIS IS THE FIX ***
    # Explicitly provide all possible labels to the report function.
    # This prevents errors if the validation split doesn't contain a sample of every class.
    print(classification_report(
        y_val,
        y_pred,
        target_names=le.classes_,
        labels=np.arange(len(le.classes_)),
        zero_division=0
    ))
    
    print("Saving model and label encoder...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lgbm, MODEL_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)
    
    print(f"\n✅ Model training complete. Model saved to '{MODEL_FILE}'")

def predict_labels():
    """Loads a trained model and predicts labels for new data."""
    print("--- Starting Prediction ---")

    print("Loading pre-trained model and label encoder...")
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
        raise FileNotFoundError("Model or label encoder not found. Please train the model first using '--mode train'.")
    
    lgbm = joblib.load(MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.csv')]

    if not test_files:
        print(f"🟡 No CSV files found in '{TEST_DIR}' to predict.")
        return

    for filename in test_files:
        try:
            print(f"   -> Processing: {filename}")
            filepath = os.path.join(TEST_DIR, filename)
            df_test = pd.read_csv(filepath)

            # Check if the dataframe is empty after loading
            if df_test.empty:
                print(f"      🟡 Skipping empty file: {filename}")
                continue
            
            # Prepare data for prediction
            X_test = df_test[FEATURES + CATEGORICAL_FEATURES].copy()
            X_test[FEATURES] = X_test[FEATURES].fillna(0)
            if 'font_name' in X_test.columns:
                X_test['font_name'] = X_test['font_name'].astype('category')

            # Make predictions
            predictions_encoded = lgbm.predict(X_test)
            predictions = le.inverse_transform(predictions_encoded)
            
            # Save results
            df_test['predicted_label'] = predictions
            output_path = os.path.join(OUTPUT_DIR, filename)
            df_test.to_csv(output_path, index=False)
            print(f"      ✅ Predictions saved to '{output_path}'")

        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"      ❌ Error reading or parsing {filename}. It might be empty or malformed. Skipping. Details: {e}")
        except KeyError as e:
            print(f"      ❌ Error processing {filename} due to a missing column: {e}. Ensure all feature columns are present. Skipping.")
        except Exception as e:
            print(f"      ❌ An unexpected error occurred with {filename}. Skipping. Details: {e}")

    print("\n✅ Prediction complete for all files.")

# --- New Main Prediction Function ---
def run_prediction(test_dir, model_dir, output_dir):
    """
    Loads a model to predict labels for CSVs in test_dir and saves results in output_dir.
    """
    print("\n--- 3. Predicting Labels ---")
    
    model_path = os.path.join(model_dir, "lgbm_model.joblib")
    encoder_path = os.path.join(model_dir, "label_encoder.joblib")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Model or encoder not found in '{model_dir}'. Please ensure they exist.")
    
    lgbm = joblib.load(model_path)
    le = joblib.load(encoder_path)
    
    os.makedirs(output_dir, exist_ok=True)
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]

    if not test_files:
        print(f"🟡 No CSV files found in '{test_dir}' to predict.")
        return

    print(f"Predicting labels for {len(test_files)} files...")
    for filename in sorted(test_files):
        try:
            print(f"   -> Predicting: {filename}")
            filepath = os.path.join(test_dir, filename)
            df_test = pd.read_csv(filepath)

            if df_test.empty:
                print(f"      🟡 Skipping empty file: {filename}")
                continue
            
            X_test = df_test[FEATURES + CATEGORICAL_FEATURES].copy()
            X_test[FEATURES] = X_test[FEATURES].fillna(0)
            if 'font_name' in X_test.columns:
                X_test['font_name'] = X_test['font_name'].astype('category')

            predictions_encoded = lgbm.predict(X_test)
            predictions = le.inverse_transform(predictions_encoded)
            
            df_test['predicted_label'] = predictions
            output_path = os.path.join(output_dir, filename)
            df_test.to_csv(output_path, index=False)

        except Exception as e:
            print(f"      ❌ An error occurred with {filename}: {e}")
    print("✅ Prediction complete.")