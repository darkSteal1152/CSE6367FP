import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from easyocr import Reader
from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
import imgproc
from collections import OrderedDict
import os
import random
from glob import glob

def show_image(title, img_bgr):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, label_prefix="Box"):
    annotated = image.copy()
    for i, box in enumerate(boxes):
        box = box.astype(np.int32)
        cv2.polylines(annotated, [box], isClosed=True, color=color, thickness=thickness)
        x, y = box[0]
        cv2.putText(annotated, f"{label_prefix} {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated

def merge_horizontal_boxes(boxes, vertical_threshold=10, horizontal_gap=20):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda b: np.mean(b[:, 1]))
    merged_boxes = []
    current_line = [boxes[0]]

    def merge_line_boxes(line_boxes):
        xs, ys = [], []
        for box in line_boxes:
            xs.extend(box[:, 0])
            ys.extend(box[:, 1])
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    for box in boxes[1:]:
        prev_box = current_line[-1]
        vertical_diff = abs(np.mean(box[:, 1]) - np.mean(prev_box[:, 1]))
        horizontal_diff = np.min(box[:, 0]) - np.max(prev_box[:, 0])
        if vertical_diff < vertical_threshold and 0 <= horizontal_diff < horizontal_gap:
            current_line.append(box)
        else:
            merged_boxes.append(merge_line_boxes(current_line))
            current_line = [box]
    if current_line:
        merged_boxes.append(merge_line_boxes(current_line))
    return merged_boxes

# CRAFT Model Initialization
print("[INFO] Loading CRAFT model...")
craft_net = CRAFT()
state_dict = torch.load('./CRAFT-pytorch/weights/craft_mlt_25k.pth', map_location='cpu')

# Modifies prefix for CRAFT training
def copyStateDict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

# Loads CRAFT Training weights
craft_net.load_state_dict(copyStateDict(state_dict))
craft_net.eval()
print("[DEBUG] CRAFT model loaded and set to eval mode.")

# Initialize EasyOCR with English Support
print("[INFO] Initializing EasyOCR...")
reader = Reader(['en'], gpu=True)

# Image Input Setup
#image_folder = r'C:\Users\Abrar Khan\Downloads\archive (1)\images'
image_folder = r'C:\Users\Abrar Khan\Downloads\archive\train_val_images\train_images'

# Finds all images
image_paths = glob(os.path.join(image_folder, '*.png')) + \
              glob(os.path.join(image_folder, '*.jpg')) + \
              glob(os.path.join(image_folder, '*.jpeg'))

# Exits if no images found
if not image_paths:
    print("[ERROR] No image files found in the folder.")
    exit()

# Image Selection
#selected_images = random.sample(image_paths, min(2, len(image_paths)))
selected_images = [r'C:\Users\Abrar Khan\Downloads\test1.png']
#selected_images = [r'C:\Users\Abrar Khan\Pictures\CSE6367FP\Figure_10.png']

# Begin Main Loop of Program
for image_path in selected_images:
    print(f"\n[PROCESSING] {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[WARNING] Could not read image: {image_path}")
        continue

    # Converts for display to show Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    show_image("Original Image", image_bgr)

    # Image Preprocessing
    canvas_size = 1280
    mag_ratio = 1.5
    # Resize image to fit in CRAFT
    resized_img, target_ratio, _ = imgproc.resize_aspect_ratio(
        image_rgb, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    # Normalize Image for CRAFT with tensor
    x = imgproc.normalizeMeanVariance(resized_img)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    # CRAFT Forward Pass
    with torch.no_grad():
        y, _ = craft_net(x)
    # CRAFT Generate Heatmaps
    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    # Optional Show Heatmaps
    '''
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(score_text, cmap='hot')
    plt.title('CRAFT Score Text Heatmap')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(score_link, cmap='hot')
    plt.title('CRAFT Link Heatmap')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    '''

    # Bounding Box Creation
    boxes, _ = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    print(f"[DEBUG] Detected {len(boxes)} text boxes.")

    # Draws Boxes on Image for optional display
    craft_annotated = draw_boxes(image_bgr, boxes, color=(255, 0, 0), label_prefix="CRAFT")
    #show_image("CRAFT Detected Boxes", craft_annotated)

    # Merging Bounding Boxes
    merged_boxes = merge_horizontal_boxes(boxes)
    print(f"[DEBUG] Merged into {len(merged_boxes)} horizontal boxes.")

    # Draws Merged Boxes on Image for optional display
    merged_annotated = draw_boxes(image_bgr, merged_boxes, color=(0, 255, 255), label_prefix="Merged")
    #show_image("Merged Boxes", merged_annotated)

    # passed to OCR
    annotated_image = image_bgr.copy()
    # For each Merged box
    for i, box in enumerate(merged_boxes):
        box = box.astype(np.int32)
        # Image cropped on box
        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])
        cropped = image_bgr[y_min:y_max, x_min:x_max]
        # Skips invalid crops
        if cropped.size == 0:
            print(f"[WARNING] Empty crop for merged box {i}")
            continue

        # EasyOCR Recognition
        result = reader.readtext(cropped)
        text = result[0][1] if result else "N/A"
        # Get Confidence Level
        confidence = result[0][2] if result else 0.0
        # Filter by confidence
        if confidence >= 0.1:
            # Print to Console
            print(f"[TEXT BOX {i}] \"{text}\" (confidence: {confidence:.2f})")

            # Annotate on Final Output
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_pos = (x_min, max(0, y_min - 10))
            cv2.putText(annotated_image, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show Final Output
    show_image("Final Annotated Image", annotated_image)
