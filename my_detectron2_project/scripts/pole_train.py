import os, json, cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

DATASET_PATH = "../pole_dataset/"
OUTPUT_DIR = "../checkpoint"
JSON_FILE_NAME = "instances_default.json"
NUM_WORKERS = 0
NUM_CLASSES = 6
BATCH_SIZE_PER_IMAGE = 512
MAX_ITER = 2000
IMS_PER_BATCH = 4
BASE_LR = 0.0002
def get_pole_dicts(img_dir):
    json_file = os.path.join(img_dir, JSON_FILE_NAME)
    with open(json_file) as f:
        dataset = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in dataset["categories"]}

    dataset_dicts = []
    for image in dataset["images"]:
        record = {}

        filename = os.path.join(img_dir, image["file_name"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = image["id"]
        record["height"] = height
        record["width"] = width

        objs = []
        for anno in dataset["annotations"]:
            if anno["image_id"] == image["id"]:
                category_name = cat_id_to_name[anno["category_id"]]

                poly = [p for xy in anno["segmentation"] for p in xy]
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": [poly],
                    "category_id": anno["category_id"],
                    "iscrowd": anno["iscrowd"]
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

for d in ["train", "test"]:
    DatasetCatalog.register("pole_" + d, lambda d=d: get_pole_dicts(DATASET_PATH + d))
    MetadataCatalog.get("pole_" + d).set(thing_classes=["background","폴리머현수", "접속개소", "LA", "TR", "폴리머LP"])
pole_metadata = MetadataCatalog.get("pole_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("pole_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.DEVICE = "cpu"
cfg.OUTPUT_DIR = OUTPUT_DIR

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
