import os
from PIL import Image
from argparse import ArgumentParser
import numpy as np
from pycocotools.coco import COCO
import logging
from pathlib import *

palette = [[254, 254, 254],[0, 0, 128],[9, 4, 15],[254, 224, 27],[0, 130, 200],[128, 0, 0],[245, 130, 48],[165, 0, 189],[57, 186, 173],[0, 89, 0]]

def files(path:str):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def coco_to_mmsegmentation(annotation_files_path: str, VERBOSE:bool) -> int:

    converted = 0
    for ann_file in files(annotation_files_path):
        print(f"Converting {ann_file}")
        ann_file_path = os.path.join(annotation_files_path, ann_file)
        folder_name = ann_file.split('_')[1].split('.')[0]
        folder_path = os.path.join(annotation_files_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        logging.info(f"Saving segmentation maps to {folder_path}")
        coco_annotations = COCO(ann_file_path)

        for image_id, image_data in coco_annotations.imgs.items():
            filename = image_data["file_name"]

            anns_ids = coco_annotations.getAnnIds(imgIds=image_id)
            image_annotations = coco_annotations.loadAnns(anns_ids)

            seg_map = np.zeros((image_data["height"], image_data["width"]), dtype=np.uint8)

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
            seg_img.save(Path(folder_path) / Path(filename).with_suffix(".png"))
            converted += 1
            print(f"Writing {filename}.png to {folder_path}") if VERBOSE else None

            # MASK FROM MMLAB CONVERTER
            mask = np.array(Image.open(os.path.join(folder_path, filename)))
            mask_copy = mask.copy()
            filename = filename.split('.')[0] + '_labelTrainIds.png'
            seg_filename = os.path.join(folder_path, filename)
            Image.fromarray(mask_copy).save(seg_filename, 'PNG')
            converted += 1
            print(f"Writing {filename} to {folder_path}") if VERBOSE else None

    return converted



def main():
    #Parse args
    parser = ArgumentParser(description="Converts datasets from COCO format to MMSegmentation format.")
    parser.add_argument('annotations_directory', metavar="ann_dir", type=str, help="Total or relative path to annotations directory")
    parser.add_argument('-v', '--verbose', action='store_const', const=True, help="Activate detailed output", default=False, required=False)
    args = parser.parse_args()
    #Call function
    try:
        num = coco_to_mmsegmentation(args.annotations_directory, args.verbose)
    except Exception as e:
        print(e)
        return
    print(f"\nConverted {num} images.")
    print("Done.")



if __name__ == "__main__":
    main()