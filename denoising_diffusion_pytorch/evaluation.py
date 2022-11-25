IOU_METRIC = "IoU"
DICE_METRIC = "Dice"
JACCARD_METRIC = "Jaccard"

def iou(predicted, ground_truth):
    return 1.0

def dice(predicted, ground_truth):
    return 0.5

def jaccard(predicted, ground_truth):
    return 0.0

EVAL_FUNCTIONS = {
    IOU_METRIC: iou,
    DICE_METRIC: dice,
    JACCARD_METRIC: jaccard
}
