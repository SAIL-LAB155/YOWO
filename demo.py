from __future__ import print_function
import cv2 as cv
import torch.nn as nn
from torchvision import datasets, transforms

import dataset
from opts import parse_opts
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss

from model import YOWO

# def letterbox(img, new_shape=(224, 224), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
#     # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
#     shape = img.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#
#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)
#
#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = new_shape
#         ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios
#
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
#
#     if shape[::-1] != new_unpad:  # resize
#         img = cv.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return img, ratio, (dw, dh)

# def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
#     # Rescale coords (xyxy) from img1_shape to img0_shape
#     if ratio_pad is None:  # calculate from img0_shape
#         gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
#         pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]
#
#     coords[:, [0, 2]] -= pad[0]  # x padding
#     coords[:, [1, 3]] -= pad[1]  # y padding
#     coords[:, :4] /= gain
#     clip_coords(coords, img0_shape)
#     return coords


def bbox_iou(box1, box2, x1y1x2y2=True):
	if x1y1x2y2:
		mx = min(box1[0], box2[0])
		Mx = max(box1[2], box2[2])
		my = min(box1[1], box2[1])
		My = max(box1[3], box2[3])
		w1 = box1[2] - box1[0]
		h1 = box1[3] - box1[1]
		w2 = box2[2] - box2[0]
		h2 = box2[3] - box2[1]
	else:
		mx = min(float(box1[0] - box1[2] / 2.0), float(box2[0] - box2[2] / 2.0))
		Mx = max(float(box1[0] + box1[2] / 2.0), float(box2[0] + box2[2] / 2.0))
		my = min(float(box1[1] - box1[3] / 2.0), float(box2[1] - box2[3] / 2.0))
		My = max(float(box1[1] + box1[3] / 2.0), float(box2[1] + box2[3] / 2.0))
		w1 = box1[2]
		h1 = box1[3]
		w2 = box2[2]
		h2 = box2[3]
	uw = Mx - mx
	uh = My - my
	cw = w1 + w2 - uw
	ch = h1 + h2 - uh
	carea = 0
	if cw <= 0 or ch <= 0:
		return 0.0

	area1 = w1 * h1
	area2 = w2 * h2
	carea = cw * ch
	uarea = area1 + area2 - carea
	return carea / uarea


def nms(boxes, nms_thresh):
	if len(boxes) == 0:
		return boxes

	det_confs = torch.zeros(len(boxes))
	for i in range(len(boxes)):
		det_confs[i] = 1 - boxes[i][4]

	_, sortIds = torch.sort(det_confs)
	out_boxes = []
	for i in range(len(boxes)):
		box_i = boxes[sortIds[i]]
		if box_i[4] > 0:
			out_boxes.append(box_i)
			for j in range(i + 1, len(boxes)):
				box_j = boxes[sortIds[j]]
				if bbox_iou(box_i, box_j, x1y1x2y2=True) > nms_thresh:
					box_j[4] = 0
	return out_boxes


def get_config():
	opt = parse_opts()  # Training settings
	dataset_use = opt.dataset  # which dataset to use
	datacfg = opt.data_cfg  # path for dataset of training and validation, e.g: cfg/ucf24.data
	cfgfile = opt.cfg_file  # path for cfg file, e.g: cfg/ucf24.cfg
	# assert dataset_use == 'ucf101-24' or dataset_use == 'jhmdb-21', 'invalid dataset'

	# loss parameters
	loss_options = parse_cfg(cfgfile)[1]
	region_loss = RegionLoss()
	anchors = loss_options['anchors'].split(',')
	region_loss.anchors = [float(i) for i in anchors]
	region_loss.num_classes = int(loss_options['classes'])
	region_loss.num_anchors = int(loss_options['num'])

	return opt, region_loss


def load_model(opt, pretrained_path):
	seed = int(time.time())
	use_cuda = True
	gpus = '0'
	torch.manual_seed(seed)
	if use_cuda:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpus
	torch.cuda.manual_seed(seed)

	# Create model
	model = YOWO(opt)
	model = model.cuda()
	# model = nn.DataParallel(model, device_ids=None)  # in multi-gpu case
	model.seen = 0

	checkpoint = torch.load(pretrained_path)
	epoch = checkpoint['epoch']
	fscore = checkpoint['fscore']
	model.load_state_dict(checkpoint['state_dict'], strict=False)

	return model, epoch, fscore


def infer(model, data, region_loss):
	num_classes = region_loss.num_classes
	anchors = region_loss.anchors
	num_anchors = region_loss.num_anchors
	conf_thresh_valid = 0.005
	nms_thresh = 0.4

	model.eval()

	data = data.cuda()
	res = []
	with torch.no_grad():
		output = model(data).data
		all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)

		for i in range(output.size(0)):
			boxes = all_boxes[i]
			boxes = nms(boxes, nms_thresh)

			for box in boxes:
				x1 = round(float(box[0] - box[2] / 2.0) * 3840.0) # same width   with the video size
				y1 = round(float(box[1] - box[3] / 2.0) * 2160.0) # same hight with the video size
				x2 = round(float(box[0] + box[2] / 2.0) * 3840.0) # same width  and hight with the video size
				y2 = round(float(box[1] + box[3] / 2.0) * 2160.0)# same width  and hight with the video size
				det_conf = float(box[4])

				for j in range((len(box) - 5) // 2):
					cls_conf = float(box[5 + 2 * j].item())
					if type(box[6 + 2 * j]) == torch.Tensor:
						cls_id = int(box[6 + 2 * j].item())
					else:
						cls_id = int(box[6 + 2 * j])
					prob = det_conf * cls_conf
					res.append(str(int(box[6]) + 1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2))
	return res


def pre_process_image(images, clip_duration, input_shape=(224, 224)):
	# resize to (224,224)
	clip = [img.resize(input_shape) for img in images]
	# clip = []
	# for img in images:
	# 	img1, ratio, (dw, dh) = letterbox(img)
	# 	clip.append(img1)
	# numpy to tensor
	op_transforms = transforms.Compose([transforms.ToTensor()])
	clip = [op_transforms(img) for img in clip]
	# change dimension
	clip = torch.cat(clip, 0).view((clip_duration, -1) + input_shape).permute(1, 0, 2, 3)
	# expand dimmension to (batch_size, channel, duration, w, h)
	clip = clip.unsqueeze(0)

	return clip


def post_process(images, bboxs):
	# ucf101_cls = ['','Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding','IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing','TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
	swim_cls = [ '', 'drown', 'swim', 'walk']
	conf_thresh = 0.1
	nms_thresh = 0.4

	proposals = []
	for i in range(len(bboxs)):
		line = bboxs[i]
		cls, score, x1, y1, x2, y2 = list(map(float, line.strip().split(' ')))
		cls = int(cls)

		if float(score) < conf_thresh:
			continue
		# proposals.append([int(int(x1) * 1920/224), int(int(y1) * 1080/224), int(int(x2) * 1920/224), int(int(y2) * 1080/224), float(score), int(cls)])
		proposals.append([int(x1), int(y1), int(x2), int(y2), float(score), int(cls)])

	proposals = nms(proposals, nms_thresh)

	image = cv.cvtColor(np.asarray(images[-1], dtype=np.uint8), cv.COLOR_RGB2BGR)
	# cv.imshow('frame', image)
	for proposal in proposals:
		x1, y1, x2, y2, score, cls = proposal
		cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2, cv.LINE_4)

		text = '[{:.2f}] {}'.format(score, swim_cls[cls])
		print(text)
		font_type = 5
		font_size = 5
		line_szie = 5
		textsize = cv.getTextSize(text, font_type, font_size, line_szie)
		y1 = y1 - 10
		p1 = (x1, y1 - textsize[0][1])
		p2 = (x1 + textsize[0][0], y1 + textsize[1])
		cv.rectangle(image, p1, p2, (180, 238, 180), -1)
		cv.putText(image, text, (x1, y1), font_type, font_size, (255, 0,0), line_szie, 2)

	return image


if __name__ == '__main__':
	duration = 8
	num_sample = 8
	pretrained_path = './backup/yowo_swim_drown_8f_checkpoint.pth'
	video_path = './data/examples/swim/drown_2.mp4'

	fourcc = cv.VideoWriter_fourcc(*'MJPG')
	out = cv.VideoWriter('output_drown_2.avi', fourcc, 5.0, (3840, 2160)) # same width  and hight with the video size

	# load parameters
	opt, region_loss = get_config()
	# load model
	model, epoch, fscore = load_model(opt, pretrained_path)
	# read video
	video = cv.VideoCapture(video_path)

	stack = []
	n = 0
	t0 = time.time()
	while True:
		ret, frame = video.read()
		if not ret:
			break
		n += 1

		frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		# cv.imshow("frame", frame)
		frame = Image.fromarray(np.uint8(frame))
		stack.append(frame)

		if len(stack) == duration:
			# 1. preprocess images
			input_data = pre_process_image(stack, duration)
			# 2. YOWO detect action tube
			output_data = infer(model, input_data, region_loss)
			# 3. draw result to images
			result_img = post_process(stack, output_data)
			# 4. write to video
			out.write(result_img)
			
			for i in range(num_sample):
				stack.pop(0)

			# t = time.time() - t0
			# print('cost {:.2f}, {:.2f} FPS'.format(t, num_sample / t))
			# t0 = time.time()

	out.release()
	video.release()
	# cv.destroyAllWindows()