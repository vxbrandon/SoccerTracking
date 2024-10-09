import json
import os
import shutil

from PIL import Image

from rich.progress import track


from src.utils.utils import list_subdirectories



def convert_bbox(bbox, img_width, img_height):
    x_center = bbox['x_center'] / img_width
    y_center = bbox['y_center'] / img_height
    width = bbox['w'] / img_width
    height = bbox['h'] / img_height
    return x_center, y_center, width, height


def convert_json_to_yolo(json_file_dir, json_file, save_dir, image_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(save_dir, exist_ok=True)

    image_dict = {img['image_id']: img for img in data['images']}

    for annotation in data['annotations']:
        if "bbox_image" not in annotation.keys():
            continue
        image_id = annotation['image_id']
        image_info = image_dict[image_id]

        img_width = image_info['width']
        img_height = image_info['height']

        bbox = annotation['bbox_image']
        x_center, y_center, width, height = convert_bbox(bbox, img_width, img_height)

        category_id = annotation['category_id'] - 1  # YOLO uses 0-indexed classes

        yolo_line = f"{category_id} {x_center} {y_center} {width} {height}"

        # Save YOLO format annotation
        yolo_filename = os.path.join(save_dir, "labels", f"{image_id}.txt")
        os.makedirs(os.path.dirname(yolo_filename), exist_ok=True)

        # Check if the line already exists
        if os.path.exists(yolo_filename):
            with open(yolo_filename, 'r') as f:
                existing_lines = f.read().splitlines()
            if yolo_line in existing_lines:
                continue  #

        # Write the line if it doesn't exist
        with open(yolo_filename, 'a') as f:
            f.write(yolo_line + '\n')

        # Copy image file to save directory
        src_image_path = os.path.join(json_file_dir, "img1", image_info['file_name'])
        dst_image_path = os.path.join(save_dir, "images", str(image_id) + ".jpg")
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        if os.path.exists(src_image_path) and not os.path.exists(dst_image_path):
            shutil.copy2(src_image_path, dst_image_path)


if __name__ == "__main__":
    split = "valid"
    save_dir = f"datasets/SNGS_yolo/{split}"  # Replace with your desired save directory
    image_dir = f"datasets/SNGS_yolo/{split}/images"

    for json_file_dir in track(list_subdirectories(f"data/SoccerNetGS/gamestate-2024/{split}"),
                               description="Converting Soccernet GS dataset into YOLO-trainable format"):
        json_filename = os.path.join(json_file_dir, "Labels-GameState.json")
        convert_json_to_yolo(json_file_dir, json_filename, save_dir, image_dir)

    print(f"Conversion complete. YOLO format files saved in {save_dir}")