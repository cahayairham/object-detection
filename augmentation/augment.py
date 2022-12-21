import random
import os
import cv2
from matplotlib import pyplot as plt

import albumentations as A
import os
import time


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_center, y_center, w, h = bbox
    image_h, image_w, _ = img.shape
    w = w * image_w
    h = h * image_h
    x_min = int(((2 * x_center * image_w) - w)/2)
    y_min = int(((2 * y_center * image_h) - h)/2)
    x_max = int(x_min + w)
    y_max = int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


dir = 'fish data/train/images'
    
for file in os.listdir(dir):
    imgname = os.fsdecode(file)
    for i in range(20):
        print('attempt: image ', imgname, 'version: ', i)
        
        # dir = "augmentation\images"
        # imgname = '00d0251'
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


        # We will use the mapping from category_id to the class name
        # to visualize the class label for the bounding box on the image
        # category_id_to_name = {17: 'cat', 18: 'dog'}


        # jpgfile = os.path.isfile(path+'.jpg')
        # pngfile = os.path.isfile(path+'.png')
        # if jpgfile:
        #     ext = '.jpg'
        # elif pngfile:
        #     ext = '.png'

        # path = path+ext

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # We will use the mapping from category_id to the class name
        # to visualize the class label for the bounding box on the image
        category_id_to_name = {0: 'Eye', 1: 'Fish'}
        # visualize(image, bboxes, category_ids, category_id_to_name)

        random.seed(round(time.time() * 1000))
        image_h, image_w, _ = image.shape
        transform = A.Compose([
                A.HorizontalFlip(p=random.randrange(0, 2)),
                A.VerticalFlip(p=random.randrange(0, 2)),
                # A.Blur(p=random.randrange(0, 2)),
                # A.ShiftScaleRotate(p=0.9),
                A.RandomBrightnessContrast(p=0.9),
                A.RGBShift(r_shift_limit=random.randrange(20, 100), 
                            g_shift_limit=random.randrange(20, 100), 
                            b_shift_limit=random.randrange(20, 100), 
                            p=0.9),
                A.RandomSizedBBoxSafeCrop(width=random.randrange(int(image_w/2), int(image_w)), 
                                            height=random.randrange(int(image_h/2), int(image_h)), 
                                            erosion_rate=random.uniform(0, 0.3),
                                            p=0.3),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ], p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.2),
                    A.Blur(blur_limit=3, p=0.2),
                ], p=0.5),
                A.OneOf([
                    # A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.3),
                    A.IAAPiecewiseAffine(p=0.3),
                ], p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                    A.RandomBrightnessContrast(),
                ], p=0.5),
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
        )

        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        # visualize(
        #     transformed['image'],
        #     transformed['bboxes'],
        #     transformed['category_ids'],
        #     category_id_to_name,
        # )

        ver=i

        cv2.imwrite('augmented_data/'+filename+str(ver)+ext, transformed['image'])
        # print('augmented_data'+dir+imgname+str(ver)+ext)

        # print(transformed['category_ids'])

        newlabel = ''
        for cat, line in zip(transformed['category_ids'], transformed['bboxes']):
            singlelabel = ''
            for num in line:
                singlelabel+=' '+str(num)
            # print('nl')
            newlabel+=str(cat)+ singlelabel+'\n'


        f = open('augmented_data/'+filename.replace("images", "labels")+str(ver)+".txt", "a")
        f.write(newlabel)
