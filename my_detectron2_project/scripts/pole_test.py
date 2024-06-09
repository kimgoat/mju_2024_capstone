import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

DATASET_PATH = "../pole_dataset/"
TEST_DATASET_PATH = "../pole_dataset/test"
OUTPUT_DIR = "./output_pole_5"
JSON_FILE_NAME = "instances_default.json"
CHECKPOINT_PATH = "../checkpoint5"
MODEL = "model_final.pth"
NUM_WORKERS = 0
NUM_CLASSES = 6
BATCH_SIZE_PER_IMAGE = 512
MAX_ITER = 2000
IMS_PER_BATCH = 4
BASE_LR = 0.0002
TEST_IMAGE_NUM = 20

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
cfg.MODEL.WEIGHTS = os.path.join(CHECKPOINT_PATH, MODEL)
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)



dataset_dicts = get_pole_dicts(TEST_DATASET_PATH)
for d in random.sample(dataset_dicts, TEST_IMAGE_NUM):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None

    print(f"Image ID: {d['image_id']}")
    print("Predicted Boxes:", boxes)
    print("Scores:", scores)
    print("Classes:", classes)

    v = Visualizer(im[:, :, ::-1], metadata=pole_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
cv2.destroyAllWindows()

evaluator = COCOEvaluator("pole_test", output_dir=OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "pole_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
