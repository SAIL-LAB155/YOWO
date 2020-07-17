import os
import shutil
import glob
import cv2

# width = 2048
# height = 1536

def modify_label(current_pth, value):
    for parent, dirname, files in os.walk(current_pth):
        for file in files:
            label_pth = os.path.join(parent, file)
            # print(label_pth)
            img_pth = label_pth.replace('labels', 'rgb-images').replace('.txt', '.jpg')
            height, width, _ = cv2.imread(img_pth).shape
            with open(label_pth, 'r') as f:
                lines = f.readlines()
                new_lines = []
                for idx, line in enumerate(lines):
                    if line != '\n':
                        c, xc, yc, w, h = list(map(float, line.strip().split(' ')))
                        new_c = str(value)
                        x1 = str((xc - 0.5*w)*width)
                        x2 = str((xc + 0.5*w)*width)
                        y1 = str((yc -0.5*h)*height)
                        y2 = str((yc + 0.5*h)*height)
                        new_line = ' '.join([new_c, x1, y1, x2, y2])+'\n' if idx != len(lines)-1 else ' '.join([new_c, x1, y1, x2, y2])
                        new_lines.append(new_line)
            with open(label_pth, 'w') as f1:
                for new_line in new_lines:
                    f1.write(new_line)

def rename_label(root, classes, dst):
    for c in classes: # walk
        label_pth = os.path.join(root, 'labels', c) #'./data/walk/labels'
        image_pth = os.path.join(root, 'rgb-images', c) #'./data/walk/rgb_images/walk'
        dirs = os.listdir(image_pth) #['walk_01', 'walk_02', ....]
        for dir in dirs:
            curr_pth = os.path.join(image_pth, dir) #'./data/walk/rgb_images/walk/walk_01'
            imgs = sorted(glob.glob(curr_pth + '/*.jpg')) #['./data/walk/rgb_images/walk/walk_01/xxx.jpg', ...]
            count = 1
            for img in imgs:
                label = os.path.join(label_pth, dir, os.path.split(img)[-1].replace('.jpg', '.txt'))
                rename_img_dir = os.path.join(dst, 'rgb-images', c, dir)
                rename_label_dir = os.path.join(dst, 'labels', c, dir)
                os.makedirs(rename_img_dir, exist_ok=True)
                os.makedirs(rename_label_dir, exist_ok=True)
                shutil.copy(img, os.path.join(rename_img_dir, str(count).zfill(5)+'.jpg'))
                shutil.copy(label, os.path.join(rename_label_dir, str(count).zfill(5)+'.txt'))
                count += 1

def generate_txt(root, cl, file_train, file_valid):
    
    label_pth = os.path.join(root, cl)
    print(label_pth)
    count = 0
    for root, dirs, files in os.walk(label_pth, topdown=False):
        
        for file in files:
            line = os.path.join(root, file) + '\n'
            if count < 10:
                with open(file_valid, 'a') as f:
                    f.write(line)
                
            else:
                with open(file_train, 'a') as f:
                    f.write(line)
            count += 1

if __name__ == '__main__':
    #modify_class(current_pth)
    # filter_w_h(current_pth)
    # rename_label(root, classes)
    root = './data/trash'
    classes = {'drown':1, 'swim':2, 'walk':3}

    file_train = './data/swim_trainlist.txt'
    file_valid = './data/swim_validlist.txt'
    dst = './data/swim_drown1'
    for k, v in classes.items():
        current_pth = os.path.join(root, 'labels', k)
        modify_label(current_pth, v)
    
    rename_label(root, list(classes.keys()), dst)
    root = './data/swim_drown1/labels'
    classes = ['drown', 'swim', 'walk']
    for cl in classes:
        generate_txt(root, cl, file_train, file_valid)
