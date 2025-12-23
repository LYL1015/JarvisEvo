from viescore import VIEScore
import json
import os
import megfile
from PIL import Image
from tqdm import tqdm
import csv
import argparse
import cv2
from torchvision.transforms import transforms
from torch import nn
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def readImage(image_path):
    return Image.open(image_path)


def process_single_item(item, vie_score,  max_retries=20):
    instruction = item['instruction']
    save_path_fullset_source_image = item['source_path']
    save_path_fullset_result_image = item['save_path']
    
    src_image_path = save_path_fullset_source_image
    save_path_item = save_path_fullset_result_image
    gt_image_path = item['gt_path']
    
    for retry in range(max_retries):
        try:
            if not os.path.exists(src_image_path) or not os.path.exists(save_path_item):
                print(f"File not found: {src_image_path} or {save_path_item}")
                return None
            pil_image_raw = readImage(megfile.smart_open(src_image_path, 'rb')).convert("RGB")
            pil_image_edited = readImage(megfile.smart_open(save_path_item, 'rb')).convert("RGB").resize((pil_image_raw.size[0], pil_image_raw.size[1]))
            text_prompt = instruction
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            file_extension = os.path.splitext(src_image_path)[1].lower()
            mime_type = mime_type_map.get(file_extension, 'image/jpeg')
            score_list = vie_score.evaluate([pil_image_raw, pil_image_edited], text_prompt, mime_type)
            sc_global, pq_global, O_score = score_list
            ## 这里简易的操作

            numpy_image_gt = cv2.cvtColor(
                np.array(readImage(gt_image_path)), cv2.COLOR_RGB2BGR
            )
            numpy_image_edit = cv2.cvtColor(
                    np.array(readImage(save_path_item)), cv2.COLOR_RGB2BGR
                )
            height, width = numpy_image_gt.shape[:2] 
            numpy_image_edit = cv2.resize(numpy_image_edit, (width, height))

            criterion_l1 = nn.L1Loss()
            criterion_l2 = nn.MSELoss()
            numpy_image_gt_ = transforms.ToTensor()(numpy_image_gt)
            numpy_image_edit_ = transforms.ToTensor()(numpy_image_edit)
            l1 = criterion_l1(numpy_image_gt_, numpy_image_edit_).detach().cpu().numpy().item()
            l2 = criterion_l2(numpy_image_gt_, numpy_image_edit_).detach().cpu().numpy().item()

            return {
                "source_image": src_image_path,
                "edited_image": save_path_item,
                "instruction": instruction,
                "data_type": save_path_item.split("/")[-2],
                "sc_global": sc_global,
                "pq_global": pq_global,
                "overall_score": O_score,
                "l1": l1,
                "l2": l2,
            }
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error processing {save_path_item} (attempt {retry + 1}/{max_retries}): {e}")
                time.sleep(3)  # Optional: wait before retrying
                print(f"Retrying {save_path_item}...")
            else:
                print(f"Failed to process {save_path_item} after {max_retries} attempts: {e}")
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_json_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./results/")
    parser.add_argument("--test_model", type=str, default="JarvisEvo")
    parser.add_argument("--backbone", type=str, default="gpt4o")
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()
    save_path_dir = args.save_path
    backbone = args.backbone
    
    with open(args.model_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    vie_score = VIEScore(backbone=backbone, key=args.api_key)

    group_list = []
    max_workers = 10
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_single_item, item, vie_score): item 
            for item in dataset
        }
        
        # Collect results
        for future in tqdm(as_completed(future_to_item), total=len(dataset), desc="Processing items"):
            try:
                result = future.result()
                if result:
                    group_list.append(result)
            except Exception as e:
                item = future_to_item[future]
                print(f"Error processing item {item.get('save_path', 'unknown')}: {e}")

    # Calculate metrics directly
    total_items = len(group_list)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total items processed: {total_items}")
    
    if total_items > 0:
        avg_sc_global = sum(item['sc_global'] for item in group_list) / total_items
        avg_pq_global = sum(item['pq_global'] for item in group_list) / total_items
        avg_overall_global = sum(item['overall_score'] for item in group_list) / total_items
        avg_l1 = sum(item['l1'] for item in group_list) / total_items
        avg_l2 = sum(item['l2'] for item in group_list) / total_items
        
        print(f"\nGlobal Metrics (all {total_items} items):")
        print(f"  Average SC: {avg_sc_global:.4f}")
        print(f"  Average PQ: {avg_pq_global:.4f}")
        print(f"  Average Overall Score: {avg_overall_global:.4f}")
        print(f"  Average L1: {avg_l1:.4f}")
        print(f"  Average L2: {avg_l2:.4f}")
