import time
from functools import wraps
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
import torch
import sys
sys.path.append("/home/user/yolov7")
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from torchvision import transforms

from .gcio import (TMP_DETECTION_OUTPUT_PATH, TMP_SEGMENTATION_OUTPUT_PATH,
                   TMP_TILS_SCORE_PATH, copy_data_to_output_folders,
                   get_image_path_from_input_folder,
                   get_tissue_mask_path_from_input_folder,
                   initialize_output_folders)
from .rw import (READING_LEVEL, WRITING_TILE_SIZE, DetectionWriter,
                 SegmentationWriter, TilsScoreWriter,
                 open_multiresolutionimage_image)


# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


@timing
def process_image_tile_to_segmentation(
    image_tile: np.ndarray, tissue_mask_tile: np.ndarray
) -> np.ndarray:
    """Example function that shows processing a tile from a multiresolution image for segmentation purposes.

    NOTE
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        image_tile (np.ndarray): [description]
        tissue_mask_tile (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """

    prediction = np.copy(image_tile[:, :, 0])
    prediction[image_tile[:, :, 0] >= 0] = 0
    return prediction * tissue_mask_tile


@timing
def process_image_tile_to_detections(
    image_tile: np.ndarray,
    segmentation_mask: np.ndarray,
    model,
    device
) -> List[tuple]:
    """Example function that shows processing a tile from a multiresolution image for detection purposes.

    NOTE
        This code is only made for illustration and is not meant to be taken as valid processing step. Please update this function

    Args:
        image_tile (np.ndarray): [description]
        tissue_mask_tile (np.ndarray): [description]

    Returns:
        List[tuple]: list of tuples (x,y) coordinates of detections
    """
    xs = []
    ys = []
    probabilities = []   

    image = transforms.ToTensor()(image_tile)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()

    output = model(image, augment=False)
    conf_thres = 0.2
    iou_thres = 0.35
    out = non_max_suppression(output[0], conf_thres, iou_thres, classes=[0])[0]
    bboxes = out[:, :4]
    nbboxes = bboxes.detach().cpu().numpy()
    
    for i in range(len(nbboxes)):
        xs.append(float((nbboxes[i][0] + nbboxes[i][2])/2))
        ys.append(float((nbboxes[i][1] + nbboxes[i][3])/2))
        probabilities.append(float(out[i][4]))

    return list(zip(xs, ys, probabilities))


@timing
def process_segmentation_detection_to_tils_score(
    segmentation_path: Path, detections: List[tuple]
) -> int:
    """Example function that shows processing a segmentation mask and corresponding detection for the computation of a tls score.

    NOTE
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        segmentation_mask (np.ndarray): [description]
        detections (List[tuple]): [description]

    Returns:
        int: til score (between 0, 100)
    """
    value = 0
    return value


def process():
    """Proceses a test slide"""

    level = READING_LEVEL
    tile_size = WRITING_TILE_SIZE  # should be a power of 2

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    print(f"Processing image: {image_path}")
    print(f"Processing with mask: {tissue_mask_path}")

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    # initialize model
    print('Setting device')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Loading weights')
    weight = torch.load('/home/user/weight/best.pt', map_location=device)
    print('Loading model')
    model = weight['model']
    model = model.half().to(device)

    # create writers
    print(f"Setting up writers")
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)

    print("Processing image...")
    # loop over image and get tiles
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()

            if not np.any(tissue_mask_tile):
                continue

            image_tile = image.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            )

            # segmentation
            segmentation_mask = process_image_tile_to_segmentation(
                image_tile=image_tile, tissue_mask_tile=tissue_mask_tile
            )
            segmentation_writer.write_segmentation(tile=segmentation_mask, x=x, y=y)

            # detection
            detections = process_image_tile_to_detections(
                image_tile=image_tile, segmentation_mask=segmentation_mask, model=model, device=device
            )
            detection_writer.write_detections(
                detections=detections, spacing=spacing, x_offset=x, y_offset=y
            )

    print("Saving...")
    # save segmentation and detection
    segmentation_writer.save()
    detection_writer.save()

    print("Number of detections", len(detection_writer.detections))

    print("Compute tils score...")
    # compute tils score
    tils_score = process_segmentation_detection_to_tils_score(
        TMP_SEGMENTATION_OUTPUT_PATH, detection_writer.detections
    )
    tils_score_writer.set_tils_score(tils_score=tils_score)

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
