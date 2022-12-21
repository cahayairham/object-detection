import cv2
import albumentations as A
import argparse
from draw_boxes import draw_boxes
import os
# construct the argument parser





parser = argparse.ArgumentParser()
parser.add_argument(
    '-f', '--format', help='bbox transform type', 
    default='yolo', choices=['coco', 'voc', 'yolo']
)
parser.add_argument(
    '-ma', '--min-area', dest='min_area', 
    help='provide value > 0 if wanting to ignore bbox after \
        augmentation under a certain threshold', 
    default=0, type=int
)
args = vars(parser.parse_args())




dir = 'augmented_data/fish data/train/images'
# dir = 'fish data/val/images'
# file = '1110.png'
    
for file in os.listdir(dir):
# if True:
    imgname = os.fsdecode(file)       
    path = os.path.join(dir, imgname)
    filename, ext = os.path.splitext(path)
    # path = dir+imgname

    bboxes = []
    category_ids = []
    with open(filename.replace("images", "labels")+'.txt') as labels:
        for line in labels:

            label = int(line[0])
            category_ids.append(label)

            coord_raw = line[1:]
            coord = []

            for num in coord_raw.split():
                coord.append(float(num))

            bboxes.append(coord)

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(imgname)



    # bbox in YOLO format
    bboxes_yolo = bboxes
    # class labels list containing all the class names
    class_labels = []
    for cat in category_ids:
        if cat==0:
            class_labels.append('Eye')
        elif cat==1:
            class_labels.append('Fish')
    print(class_labels)

    if args['format'] == 'yolo':
        transform = A.Compose([
            ], bbox_params=A.BboxParams(
            format='yolo', label_fields=['class_labels'],
            min_area=args['min_area']
        ))
        transformed_instance = transform(
            image=image, bboxes=bboxes_yolo, class_labels=class_labels
        )
        transformed_image = transformed_instance['image']
        transformed_bboxes = transformed_instance['bboxes']
    # draw the bounding boxes on the tranformed/augmented image
    annot_image, box_areas = draw_boxes(
        transformed_image, transformed_bboxes, class_labels, args['format']
    )
    print('augmented_data/augmented_data_with_bbox/'+os.path.splitext(file)[0]+'bbox'+ext)
    # cv2.imshow('Image', annot_image)
    cv2.imwrite('augmented_data/augmented_data_with_bbox/'+os.path.splitext(file)[0]+'bbox'+ext, annot_image)

    cv2.waitKey(0)