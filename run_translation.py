# -*- coding: utf-8 -*-

# Generic
import os
import yaml
import string
from shutil import copyfile
import warnings; warnings.filterwarnings("ignore", category=UserWarning)

# Img processing
from PIL import ImageOps, Image

# Pytorch
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# Text recognition imports
from text_recognition.utils import CTCLabelConverter, AttnLabelConverter
from text_recognition.dataset import TabularDataset, AlignCollate

# Inpainting
from mmedit.apis import init_model, inpainting_inference
from mmedit.core import tensor2img

# Custom files and classes
from utils import *
from bounding_box import *
from custom_text import *

if __name__ == '__main__':

    # read hyper parameters

    config = None
    try:
        with open("config.yaml", 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # GPU config

    device = torch.device('cuda' if torch.cuda.is_available() and config["cuda"] else 'cpu')
    if (device == "cuda"):
        ## number of gpus
        num_gpu = torch.cuda.device_count()
        # CuDNN config : As background for CuDNN for many operations, CuDNN has several implementations
        ## to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware
        cudnn.benchmark = True
        ## allow only cudnn determinstic implementations/algos
        cudnn.deterministic = True

    # Preparing inputs and result output

    ## get test images names from the folder
    image_list, _, _ = file_utils.get_files(config["data_folder"])
    ## prepare result folder
    result_folder = config["results_folder"]
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    # load and prepare our text detector
    ## Load and prepare our text detector
    craft = model_builder(config, device, model_type="text_detector", inference=True)
    poly = config["text_detector"]["poly"]
    ## Do same for the LinkRefiner Network
    refine_net = None
    if config["text_detector"]["refine"]:
        poly = True
        refine_net = model_builder(config, device, model_type="refiner", inference=True)

    # load and prepare our text recognizer
    ## vocab / character number configuration
    if config["text_recognizer"]["sensitive"]:
        config["text_recognizer"]["character"] = string.printable[:-6]  # same with ASTER setting (use 94 char).
    ## converter
    if config["text_recognizer"]["Prediction"] == 'CTC':
        converter = CTCLabelConverter(config["text_recognizer"]["character"])
    else:
        converter = AttnLabelConverter(config["text_recognizer"]["character"])
    ## number of classes
    config["text_recognizer"]["num_class"] = len(converter.character)
    ## rgb images
    if config["text_recognizer"]["rgb"]:
        config["text_recognizer"]["input_channel"] = 3
    ## load model
    STR = model_builder(config, device, model_type="text_recognizer", inference=True)

    # Process our images Read ==> Predict Boxes ==> Predict words
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path))
        image = imgproc.loadImage(image_path)
        # rotate image to adjust the text horizontally
        # image = imgproc.rotateImage(image)
        with torch.no_grad():

            # text detection
            boxes = text_det_infer(craft, refine_net, poly, config, image, device, verbose=True)
            # file_utils.saveResult(image_path, image[:,:,::-1], boxes, dirname=result_folder)

            t0 = time.time()
            # text recognition
            boxes_data = []
            boundaries = []
            for i, box in enumerate(boxes):
                box = box.astype(int)
                x1, x2 = box[0, 1], box[2, 1]
                y1, y2 = box[0, 0], box[1, 0]
                boxes_data.append(image[x1: x2, y1: y2])
                boundaries.append(box.ravel())

            AlignCollate_boxes = AlignCollate(imgH=config["text_recognizer"]["imgH"],
                                              imgW=config["text_recognizer"]["imgW"],
                                              keep_ratio_with_pad=config["text_recognizer"]["pad"])

            boxes_data = TabularDataset(data=boxes_data,
                                        bboxes=boundaries,
                                        opt=objectview(config["text_recognizer"]))
            boxes_loader = torch.utils.data.DataLoader(boxes_data,
                                                       batch_size=config["text_recognizer"]["batch_size"],
                                                       shuffle=False,
                                                       num_workers=int(config["text_recognizer"]["workers"]),
                                                       collate_fn=AlignCollate_boxes, pin_memory=True)

            words_crops_batch, boxes_batch = next(iter(boxes_loader))
            batch_size = words_crops_batch.size(0)
            words_crops_batch = words_crops_batch.to(device)

            len_for_predictions = torch.IntTensor([config["text_recognizer"]["max_label_len"]] *
                                                  batch_size).to(device)
            text_for_predictions = torch.LongTensor(1, config["text_recognizer"]["max_label_len"] + 1). \
                fill_(0).to(device)

            if 'CTC' in config["text_recognizer"]["Prediction"]:
                predictions = STR(words_crops_batch, text_for_predictions)
                # Select max probability (greedy decoding) then decode index to character
                predictions_size = torch.IntTensor([predictions.size(1)] * batch_size)
                _, predictions_index = predictions.max(2)
                predictions_str = converter.decode(predictions_index, predictions_size)

            else:
                predictions = STR(words_crops_batch, text_for_predictions, is_train=False)
                # select max probabilty (greedy decoding) then decode index to character
                _, predictions_index = predictions.max(2)
                predictions_str = converter.decode(predictions_index, len_for_predictions)

            predictions_prob = F.softmax(predictions, dim=2)
            predictions_max_prob, _ = predictions_prob.max(dim=2)

            b_boxes_manager = BoundingBoxesManager([])
            for box, predictions, predictions_max_prob in zip(boxes_batch, predictions_str, predictions_max_prob):
                if 'Attn' in config["text_recognizer"]["Prediction"]:
                    predictions_EOS = predictions.find('[s]')
                    predictions = predictions[:predictions_EOS]  # prune after "end of sentence" token ([s])
                    predictions_max_prob = predictions_max_prob[:predictions_EOS]

                # detect the text color
                box_h, box_w = int(box[7] - box[1]), int(box[2] - box[0])
                box_crop_arr = image[box[1]:box[1] + box_h, box[0]:box[0] + box_w].astype(np.uint8)
                mask_text = np.array(ImageOps.invert(Image.fromarray(box_crop_arr).convert("L"))).astype(np.uint8) > 80
                box_crop_arr, mask_text = box_crop_arr.reshape(-1, 3), mask_text.reshape(-1)
                color = tuple(np.median(box_crop_arr[mask_text,], axis=0).astype(np.int16))
                # import collections
                # color=collections.Counter(list(map(tuple, box_crop_arr[mask_text,]))).most_common(1)[0][0]
                # print("---------------------------------------------------------------")

                custom_text = CustomText(content=predictions,
                                         font_file=config["font_file_path"],
                                         font_size=box_h,
                                         color=color)

                b_boxes_manager.add(BoundingBox(box.reshape(4, 2), custom_text=custom_text))
                # calculate confidence score (= multiply of pred_max_prob)
                # confidence_score = predictions_max_prob.cumprod(dim=0)[-1]

            print("Recognition time : {:.3f}".format(time.time() - t0))

            t0 = time.time()
            b_boxes_manager.merge(word_th=config["merge_box_th"], same_line_th=config["same_line_th"])
            # b_boxes_manager.display()
            # save bounding boxes results
            b_boxes_manager.save(image_path, image, dir_name=config["results_folder"])
            print("Merging time : {:.3f}".format(time.time() - t0))

            # result path
            result_filename = "res_" + os.path.basename(image_path)
            result_path = os.path.join(result_folder, result_filename)

            # detect source language to see if it s worth it to continue or not in case we want to translate it
            # to the same original language
            src_language = config["src_lang"]
            dst_language = config["dest_lang"]

            if not config["src_lang"]:
                src_language = b_boxes_manager.detect_src_lang(translation_service=
                                                               config["translation_service"])
            if src_language == dst_language:
                copyfile(image_path, result_path)
            else:
                torch.cuda.empty_cache()
                # Inpainting
                t0 = time.time()
                ## create mask for inpainting
                mask = b_boxes_manager.mask(image)
                ## Save temp mask file temporarly
                mask_path = "temp.png"
                Image.fromarray(mask).save("temp.png")
                ## init the inpainter
                model = init_model(config["Inpainter"]["configuration"],
                                   config["Inpainter"]["pretrained_model"],
                                   device=device)
                ## inpaint
                result = inpainting_inference(model, image_path, mask_path)[0]
                result = tensor2img(result, min_max=(-1, 1))
                os.remove("temp.png")
                print("Inpainting time : {:.3f}".format(time.time() - t0))

                # Translation
                t0 = time.time()
                result = b_boxes_manager.fill_translation(result,
                                                          same_block_th=config["same_block_th"],
                                                          translation_service=config["translation_service"],
                                                          src_language=src_language,
                                                          dest_language=dst_language,
                                                          result_path=result_path)
                print("Translation time : {:.3f}".format(time.time() - t0))

                # open results
                # Image.open(image_path).show(title="source")
                # Image.open(result_path).show(title="translated")

