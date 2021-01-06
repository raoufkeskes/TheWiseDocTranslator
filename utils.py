import time
import cv2
from collections import OrderedDict

import torch
import numpy as np

from text_detection import craft_utils
from text_detection import imgproc
from text_detection.craft import CRAFT
from text_detection.refinenet import RefineNet
from text_recognition.model import Model as STR

class objectview(object):
    """
    to convert Yaml object from a dictionary to an object for more readability
    where we could type config.hello_world instead of config["hello_world"]
    """
    def __init__(self, d):
        self.__dict__ = d

def copyStateDict(state_dict):
    """
    to load the weights dictionary
    :param state_dict: the Network state dictionary in a specific format : dict
    :return: state_dict
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def model_builder(config, device, model_type="text_detector", inference=True):
    """
    to build a model whether it is text detector or recognizer
    :param config: (python yaml object) containing the configuration of the entire application
    :param device: (str) 'cpu' or 'cuda'
    :param model_type: (str) 'text_detector' or 'text_recognizer'
    :param inference: (Boolean) to set the model in evaluation mode (BatchNorm, Dropoout, ...)
    :return: (torch.nn) Model object
    """

    model, loader, weights = None, None, None
    # initialize net archi
    if model_type == "text_detector":
        model = CRAFT().to(device)
        weights = config["text_detector"]["pretrained_model"]
        loader = copyStateDict(torch.load(weights, map_location=device))
        # loading weights
        # print('Loading {} weights from checkpoint {}'.format(model_type, weights))
        model.load_state_dict(loader)
        model = torch.nn.DataParallel(model)
    elif model_type == "refiner":
        model = RefineNet().to(device)
        weights = config["text_detector"]["refiner_model"]
        loader = copyStateDict(torch.load(weights, map_location=device))
        # loading weights
        print('Loading {} weights from checkpoint {}'.format(model_type, weights))
        # model.load_state_dict(loader)
        model = torch.nn.DataParallel(model)
    elif model_type == "text_recognizer":
        opt = objectview(config["text_recognizer"])
        model = STR(opt)
        model = torch.nn.DataParallel(model.to(device))
        weights = opt.pretrained_model
        loader = torch.load(weights, map_location=device)
        # loading weights
        # print('Loading {} weights from checkpoint {}'.format(model_type, weights))
        model.load_state_dict(loader)

    if inference:
        # eval to make the BatchNorm work correctly in test mode
        model.eval()

    return model


def text_det_infer(model, refiner, poly, config, image, device, verbose=True):
    """
    :param model: (torch.nn) the neural net object
    :param refiner: (torch.nn) another model to bind words together (not necessary in our case)
    :param poly: (Boolean)
    :param config: (python yaml object) containing the configuration of the entire application
    :param image: (numpy array) of shape [h, w, c]
    :param device: (str) 'cpu' or 'cuda'
    :param verbose: (boolean) if you want to print verbose messages
    :return: (numpy arrays)  of shape [number of detected words, 4, 2] bounding boxes
    """
    t0 = time.time()
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image,
                                                                          config["text_detector"]["canvas_size"],
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=config["text_detector"]["mag_ratio"])
    ratio_h = ratio_w = 1 / target_ratio


    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]
    if device == "cuda" :
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = model(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refiner is not None:
        with torch.no_grad():
            y_refiner = refiner(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link,
                                           config["text_detector"]["text_threshold"],
                                           config["text_detector"]["link_threshold"],
                                           config["text_detector"]["low_text"],
                                           poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    # for k in range(len(polys)):
    #     if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t0

    # render results (optional)
    # render_img = score_text.copy()
    # render_img = np.hstack((render_img, score_link))
    # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if verbose : print("Detection time : {:.3f}".format(t1))

    return boxes
