import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

# Read from the cleaned CSV
df = pd.read_csv("data/annotated_reviews.csv", encoding="utf-8-sig", keep_default_na=False)

# Drop rows with no images
df = df[df['image_paths'].astype(str).str.strip() != ''].reset_index(drop=True)

# --- Balanced stratified split ---
train_frames, eval_frames = [], []
for label in [0, 1, 2]:
    cls = df[df['label'] == label].sample(frac=1, random_state=42)
    train_frames.append(cls.iloc[:80])
    eval_frames.append(cls.iloc[80:100])

train_df = pd.concat(train_frames).sample(frac=1, random_state=42).reset_index(drop=True)
eval_df  = pd.concat(eval_frames).sample(frac=1, random_state=42).reset_index(drop=True)

train_df.to_csv("data/training.csv", index=False, encoding="utf-8-sig")
eval_df.to_csv("data/evaluate.csv",  index=False, encoding="utf-8-sig")

print(f"Training: {len(train_df)} rows | Eval: {len(eval_df)} rows")
print("Training labels:\n", train_df['label'].value_counts())
print("Eval labels:\n", eval_df['label'].value_counts())

# --- Download images ---
output_dir = "data/images"
os.makedirs(output_dir, exist_ok=True)

all_rows = pd.concat([train_df, eval_df])

for idx, row in all_rows.iterrows():
    for filename in [f.strip() for f in str(row['image_paths']).split(';')]:
        if not filename or filename.lower() in ('nan', 'none'):
            continue
        save_path = os.path.join(output_dir, filename)
        if os.path.exists(save_path):
            continue
        url = f"https://down-ph.img.susercontent.com/file/{filename.replace('.jpg', '')}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(save_path, "JPEG")
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Failed {filename}: {e}")