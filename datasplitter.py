import os
import shutil
import random
import glob

# --- CONFIG ---
SOURCE_ROOT = "judo_datasetDONTDELETE"  # Your tool's output folder
DEST_ROOT = "datasets/judo_pose"        # The folder YOLO will actually use
TRAIN_RATIO = 0.8                       # 80% Training, 20% Validation

def clamp(val):
    """Restricts normalized coordinates to 0.0-1.0"""
    return max(0.0, min(1.0, val))

def clean_and_copy(src_txt, dst_txt):
    """
    Reads a YOLO label file, clamps coordinates, 
    but PRESERVES visibility flags (0, 1, 2).
    """
    with open(src_txt, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        
        # 1. Class ID
        class_id = int(parts[0])
        
        # 2. Bounding Box (4 values) - Clamp these
        bbox = [clamp(x) for x in parts[1:5]]
        
        # 3. Keypoints (Remaining values)
        # Format: x, y, visibility, x, y, visibility...
        raw_kpts = parts[5:]
        cleaned_kpts = []
        
        for i in range(0, len(raw_kpts), 3):
            kx = clamp(raw_kpts[i])     # Clamp X
            ky = clamp(raw_kpts[i+1])   # Clamp Y
            kv = int(raw_kpts[i+2])     # DO NOT CLAMP VISIBILITY (Keep 2 as 2)
            cleaned_kpts.extend([kx, ky, kv])
            
        # Reconstruct the line
        final_data = [class_id] + bbox + cleaned_kpts
        cleaned_lines.append(" ".join(map(str, final_data)))
        
    with open(dst_txt, 'w') as f:
        f.write("\n".join(cleaned_lines))

def main():
    print(f"🚀 Starting Data Split from: {SOURCE_ROOT}")

    # 1. Clear old dataset to avoid duplicates
    if os.path.exists(DEST_ROOT):
        shutil.rmtree(DEST_ROOT)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DEST_ROOT, split, 'labels'), exist_ok=True)

    # 2. Gather files (Fixed GLOB to look deeper)
    # Looks for: Root / VideoName / pose / images / *.jpg
    all_images = glob.glob(os.path.join(SOURCE_ROOT, "*", "*", "images", "*.jpg"))
    
    # Fallback: Try the old shallow path just in case
    if not all_images:
        all_images = glob.glob(os.path.join(SOURCE_ROOT, "*", "images", "*.jpg"))

    if not all_images:
        print("❌ Error: No images found! Check your directory structure.")
        return

    print(f"Found {len(all_images)} total images.")

    pairs = []
    for img_path in all_images:
        # Construct label path based on image path
        # Replaces .../images/name.jpg with .../labels/name.txt
        label_path = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace(".jpg", ".txt")
        
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
        else:
            print(f"⚠️ Warning: Missing label for {os.path.basename(img_path)}")
           
    if not pairs:
        print("❌ Error: Found images but no matching .txt labels.")
        return

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
            
            # Copy Image
            shutil.copy(img, os.path.join(DEST_ROOT, name, 'images', fname))
            
            # Clean & Copy Label
            clean_and_copy(lbl, os.path.join(DEST_ROOT, name, 'labels', fname.replace('.jpg', '.txt')))

    print(f"✅ Ready for training! Data is in: {DEST_ROOT}")

if __name__ == "__main__":
    main()