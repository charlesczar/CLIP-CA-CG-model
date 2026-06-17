import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

train_df = pd.read_csv("data/training.csv", encoding="utf-8-sig")
eval_df  = pd.read_csv("data/evaluate.csv",  encoding="utf-8-sig")

output_dir = "data/images"
os.makedirs(output_dir, exist_ok=True)

all_rows = pd.concat([train_df, eval_df])

for idx, row in all_rows.iterrows():
    if not row['image_paths']:
        continue
    for url in [u.strip() for u in str(row['image_paths']).split('|')]:
        filename = url.split('/file/')[-1]
        save_path = os.path.join(output_dir, filename + ".jpg")
        if os.path.exists(save_path):
            continue
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(save_path, "JPEG")
            print(f"Saved: {filename}.jpg")
        except Exception as e:
            print(f"Failed {filename}: {e}")