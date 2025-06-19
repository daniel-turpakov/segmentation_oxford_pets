import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from typing import Optional
import tqdm
import os
import xml.etree.ElementTree as ET
import cv2

from dataset import PetDataset
from utils import calculate_iou


def segment_with_sam(predictor: SamPredictor, image: np.ndarray, input_point: Optional[tuple[int, int]] = None) -> np.ndarray:
    """
    Получает сегментационную маску с помощью SAM

    :param predictor: модель SamPredictor
    :param image: изображение
    :param input_point: точка на изображении
    :return masks: сегментационная маска
    """
    predictor.set_image(image)

    if input_point is not None:
        input_point = np.array([[input_point[0], input_point[1]]])
        input_label = np.array([1])
        masks, _, _  = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
    else:
        masks, _, _ = predictor.predict(
            multimask_output=False
        )
    return masks.cpu().numpy()


def generate_refined_segmaps(predictor: SamPredictor, dataset: torch.utils.data.Dataset, xml_dir: str, output_dir: str) -> bool:
    """
    Уточняет сегментационную маску с помощью SAM

    :param predictor: модель SamPredictor
    :param dataset: датасет
    :param xml_dir: путь к xml данным
    :param output_dir: путь к выходной директории

    :return: bool
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm.tqdm(range(len(dataset))):
        image, gt_mask = dataset[i]
        image_name = dataset.image_names[i]

        path = os.path.join(xml_dir, f"{image_name}.xml")
        path_exists = os.path.exists(path)
        if path_exists:
            tree = ET.parse(path)
            root = tree.getroot()
            bbox = [0, 0, 0, 0] # xmin, ymin, xmax, ymax
            for i in range(4):
                bbox[i] = int(root[-1][-2][i].text) # root[-1][-2] соответствует annotations -> object -> bndbox

            mean_point = ((bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2)

            sam_map = segment_with_sam(predictor, image, mean_point).astype(np.uint8)

            if not image_name[0].isupper():
                sam_map = sam_map * 2

        write_path = os.path.join(output_dir, f"{image_name}.png")
        out_mask = sam_map[0, ..., None] if path_exists and calculate_iou(sam_map[0], gt_mask) > 0.7 else gt_mask
        if not cv2.imwrite(write_path, out_mask):
            raise Exception("Could not write image")

    return True


if __name__ == '__main__':
    data_root = "data"
    dataset = PetDataset(data_root)

    sam_checkpoint = "sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    generate_refined_segmaps(predictor, dataset, os.path.join(data_root, "annotations", "xmls"),
                             output_dir=os.path.join(data_root, "annotations", "sam_output"))
