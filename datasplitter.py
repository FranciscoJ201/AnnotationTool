import os
import shutil
import random
import glob

# --- CONFIG ---
SOURCE_ROOT = "judo_dataset"         # Your tool's output folder
DEST_ROOT = "datasets/judo_pose"     # The folder YOLO will actually use
TRAIN_RATIO = 0.8                    # 80% Training, 20% Validation

def clamp(val):
    return max(0.0, min(1.0, val))

def clean_and_copy(src_txt, dst_txt):
    """Clamps values to 0-1 and saves."""
    with open(src_txt, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        # Clamp Bbox and Keypoints
        new_parts = [int(parts[0])] + [clamp(x) for x in parts[1:]]
        # Restore Visibility flags (every 3rd keypoint value) to integers if needed
        # (Usually clamping float visibility 2.0 -> 1.0 is fine for float, but let's keep it simple)
        cleaned_lines.append(" ".join(map(str, new_parts)))
        
    with open(dst_txt, 'w') as f:
        f.write("\n".join(cleaned_lines))

def main():
    # 1. Clear old dataset to avoid duplicates
    if os.path.exists(DEST_ROOT):
        shutil.rmtree(DEST_ROOT)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DEST_ROOT, split, 'labels'), exist_ok=True)

    # 2. Gather files
    all_images = glob.glob(os.path.join(SOURCE_ROOT, "*", "images", "*.jpg"))
    pairs = []
    for img_path in all_images:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
    
    # 3. Split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_set = pairs[:split_idx]
    val_set = pairs[split_idx:]

    # 4. Copy
    for subset, name in [(train_set, "train"), (val_set, "val")]:
        print(f"Processing {name} ({len(subset)} images)...")
        for img, lbl in subset:
            fname = os.path.basename(img)
            shutil.copy(img, os.path.join(DEST_ROOT, name, 'images', fname))
            clean_and_copy(lbl, os.path.join(DEST_ROOT, name, 'labels', fname.replace('.jpg', '.txt')))

    print(f"✅ Ready for training! Data is in: {DEST_ROOT}")

if __name__ == "__main__":
    main()
    print('done')