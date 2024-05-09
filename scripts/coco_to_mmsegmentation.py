import os.path as osp
from PIL import Image
import cv2
import numpy as np
from pycocotools.coco import COCO
import logging
import json
from pathlib import *

ann_src = "/home/rodenni/mmseg/Dataset/Original/annotations/instances_train.json"
ann_dst = "/home/rodenni/mmseg/Dataset/Original/Converted/annotations/annotations.txt"
msk_dst = "/home/rodenni/mmseg/Dataset/Original/Converted/masks/"

palette = [[0, 0, 128],[9, 4, 15],[254, 224, 27],[0, 130, 200],[128, 0, 0],[245, 130, 48],[165, 0, 189],[57, 186, 173],[0, 89, 0]]

def coco_to_mmsegmentation(
    annotations_file: str, output_annotations_file: str, output_masks_dir: str
):
    """
    Args:
        annotations_file:
            path to json in [segmentation format](https://gradiant.github.io/ai-dataset-template/supported_tasks/#segmentation)
        output_annotations_file:
             path to write the txt in [mmsegmentation format](https://mmsegmentation.readthedocs.io/en/latest/tutorials/customize_datasets.html#customize-datasets-by-reorganizing-data)
        output_masks_dir:
            path where the masks generated from the annotations will be saved to.
            A single `{file_name}.png` mask will be generated for each image.
    """

    Path(output_annotations_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_masks_dir).mkdir(parents=True, exist_ok=True)
    # (ann_dst_pth, ann_dst_fn) = os.path.split(output_annotations_file)
    # os.makedirs(ann_dst_pth, exist_ok=True)
    # os.makedirs(output_masks_dir, exist_ok=True)

    logging.info(f"Loading annotations form {annotations_file}")
    annotations = json.load(open(annotations_file))

    logging.info(f"Saving annotations to {output_annotations_file}")
    with open(output_annotations_file, "w") as f:
        for image in annotations["images"]:
            filename = Path(image["file_name"]).parent / Path(image["file_name"]).stem
            f.write(str(filename))
            f.write("\n")

    logging.info(f"Saving masks to {output_masks_dir}")
    coco_annotations = COCO(annotations_file)
    for image_id, image_data in coco_annotations.imgs.items():
        filename = image_data["file_name"]
        print(filename)

        anns_ids = coco_annotations.getAnnIds(imgIds=image_id)
        image_annotations = coco_annotations.loadAnns(anns_ids)

        seg_map = np.zeros(
            (image_data["height"], image_data["width"]), dtype=np.uint8
        )
        for image_annotation in image_annotations:
            category_id = image_annotation["category_id"]
            try:
                category_mask = coco_annotations.annToMask(image_annotation)
            except Exception as e:
                logging.warning(e)
                logging.warning(f"Skipping {image_annotation}")
                continue
            category_mask *= category_id
            category_mask *= seg_map == 0
            seg_map += category_mask

        seg_img = Image.fromarray(seg_map).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(Path(output_masks_dir) / Path(filename).with_suffix(".png"))

        # output_filename = Path(output_masks_dir) / Path(filename).with_suffix(".png")
        # output_filename.parent.mkdir(parents=True, exist_ok=True)

        # logging.info(f"Writting mask to {output_filename}")
        # cv2.imwrite(str(output_filename), seg_map)
        # break
            

coco_to_mmsegmentation(ann_src, ann_dst, msk_dst)