import cv2
import json
import numpy as np
from tqdm import tqdm
from shapely.wkt import loads
import config
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

DAMAGE_DICT = {
    "un-classified": 1, "no-damage": 1,
    "minor-damage": 2, "major-damage": 3, "destroyed": 4
}

def create_masks(json_path, img_shape=(1024, 1024)):
    mask = np.zeros(img_shape, dtype=np.uint8)
    edge_mask = np.zeros(img_shape, dtype=np.uint8)

    # Fail-safe in case a test image is missing its JSON label
    if not os.path.exists(json_path):
        return mask, edge_mask

    with open(json_path, 'r') as f:
        data = json.load(f)

    for feature in data['features']['xy']:
        poly_wkt = feature['wkt']
        poly = loads(poly_wkt)
        coords = np.array(poly.exterior.coords, dtype=np.int32)

        damage = feature['properties'].get('subtype', 'no-damage')
        val = DAMAGE_DICT.get(damage, 1)

        cv2.fillPoly(mask, [coords], val)
        cv2.polylines(edge_mask, [coords], isClosed=True, color=1, thickness=2)

    return mask, edge_mask

def process_directory(raw_dir, processed_dir, is_train=True):
    images_dir = os.path.join(raw_dir, 'images')
    labels_dir = os.path.join(raw_dir, 'labels')

    if not os.path.exists(images_dir):
        print(f"Skipping... Directory not found: {images_dir}")
        return

    files = [f for f in os.listdir(images_dir) if '_post_disaster' in f]

    print(f"Preprocessing data from {raw_dir} -> {processed_dir}")
    for post_img_name in tqdm(files):
        pre_img_name = post_img_name.replace('_post_disaster', '_pre_disaster')

        pre_img = cv2.imread(os.path.join(images_dir, pre_img_name))
        post_img = cv2.imread(os.path.join(images_dir, post_img_name))

        post_json = os.path.join(labels_dir, post_img_name.replace('.tif', '.json'))
        mask, edge_mask = create_masks(post_json)

        # Save Global Downsampled Views
        global_pre = cv2.resize(pre_img, (config.TILE_SIZE, config.TILE_SIZE))
        global_post = cv2.resize(post_img, (config.TILE_SIZE, config.TILE_SIZE))
        base_name = post_img_name.split('_post')[0]

        cv2.imwrite(os.path.join(processed_dir, f"{base_name}_global_pre.png"), global_pre)
        cv2.imwrite(os.path.join(processed_dir, f"{base_name}_global_post.png"), global_post)

        # Crop into 256x256 Tiles
        stride = config.TILE_SIZE
        for y in range(0, 1024, stride):
            for x in range(0, 1024, stride):
                tile_mask = mask[y:y+stride, x:x+stride]

                # Balance background ONLY for training data. Test data must keep all tiles.
                if is_train and np.max(tile_mask) == 0 and np.random.rand() > 0.1:
                    continue

                tile_pre = pre_img[y:y+stride, x:x+stride]
                tile_post = post_img[y:y+stride, x:x+stride]
                tile_edge = edge_mask[y:y+stride, x:x+stride]

                tile_name = f"{base_name}_{y}_{x}"
                cv2.imwrite(os.path.join(processed_dir, f"{tile_name}_pre.png"), tile_pre)
                cv2.imwrite(os.path.join(processed_dir, f"{tile_name}_post.png"), tile_post)
                cv2.imwrite(os.path.join(processed_dir, f"{tile_name}_mask.png"), tile_mask)
                cv2.imwrite(os.path.join(processed_dir, f"{tile_name}_edge.png"), tile_edge)

if __name__ == '__main__':
    print("--- 1. Processing Training Data (tier1) ---")
    process_directory(config.RAW_TRAIN_DIR, config.PROCESSED_TRAIN_DIR, is_train=True)

    print("\n--- 2. Processing Testing Data (test) ---")
    process_directory(config.RAW_TEST_DIR, config.PROCESSED_TEST_DIR, is_train=False)