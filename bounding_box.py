from text_detection import file_utils
import numpy as np
from googletrans import Translator
from PIL import Image, ImageFont, ImageDraw
from textblob import TextBlob, exceptions



class BoundingBox:
    """
    a class representing the boudning boxes
    """

    def __init__(self, arr, custom_text):
        """
        :param arr: (numpy array) of shape [4, 2]  coordinates of 4 edges
        :param custom_text: (custom_text object) representing the content of the box
        """
        self.p1, self.p2, self.p3, self.p4 = arr
        self.custom_text = custom_text
        self.merged = False
        self.rectanglize()

    def rectanglize(self):
        """
        a method forcing the bounding box to be a rectangle
        """
        self.p1 = min(self.p1[0], self.p4[0]), min(self.p1[1], self.p2[1])
        self.p2 = max(self.p2[0], self.p3[0]), self.p1[1]
        self.p3 = self.p2[0], max(self.p3[1], self.p4[1])
        self.p4 = self.p1[0], self.p3[1]

    def is_closer_h(self, box2, word_th=10, same_line_th=6):
        """
        a method testing if two bounding boxes are closer horizontally (row/line wise)
        :param box2: (bounding box object)
        :param word_th: (int) representing the number pixels separating two boxes
                              horizontally to consider them as closer
        :param same_line_th: (int) representing the pixels separating two boxes
                             horizontally to consider them in the same line
        :return: (Boolean)
        """
        # Make sure that box1 is before box2 (line-wise)
        prec_b, next_b = self, box2
        if self.p1[0] > box2.p1[0]:
            prec_b, next_b = box2, self

        same_line = abs(prec_b.p1[1] - next_b.p1[1]) <= same_line_th or \
                    abs(prec_b.p4[1] - next_b.p4[1]) <= same_line_th

        overlapped = (prec_b.p2[0] >= next_b.p1[0])
        closed = (next_b.p1[0] - prec_b.p2[0] <= word_th)
        if same_line and (overlapped or closed):
            return True
        return False

    def is_closer_v(self, box2, same_block_th):
        """
        :param box2: (bounding box object)
        :param same_block_th: (int) representing the number pixels separating two boxes
                              vertically to consider them as closer to fine tune the font size
        :return: (Boolean)
        """
        return (abs(self.p4[1] - box2.p1[1]) <= same_block_th) or \
               (abs(self.p1[1] - box2.p4[1]) <= same_block_th)

    def refine_font_size(self, box2):
        """
        adjust font size
        :param box2: (bounding box object)
        """
        self.custom_text.refine_font_size(box2.custom_text)

    def merge(self, box2, sep=" "):
        """
        merge two bounding boxes (self, other) to self
        :param box2: (bounding box object)
        :param sep: (str)
        """
        self.p1 = min(self.p1[0], box2.p1[0]), min(self.p1[1], box2.p1[1])
        self.p2 = max(self.p2[0], box2.p2[0]), self.p1[1]
        self.p3 = self.p2[0], max(self.p3[1], box2.p3[1])
        self.p4 = self.p1[0], self.p3[1]

        self.custom_text.append(is_precedent=(self.p1[0] < box2.p1[0]),
                                sep=sep,
                                custom_text2=box2.custom_text)

    def is_merged(self):
        return self.merged

    def set_merged(self):
        self.merged = True

    def to_array(self):
        """
        :return: (numpy array) of shape [4, 2]
        """
        return np.array([self.p1, self.p2, self.p3, self.p4])

    def get_width_height(self):
        """
        :return: (int, int) width, height of the bounding boxes
        """
        return self.p2[0] - self.p1[0], self.p4[1] - self.p1[1]

    def __str__(self):
        return "{} \t {}".format(self.to_array().ravel(), self.custom_text.content)


class BoundingBoxesManager:
    """
    a class to manage a list of bounding boxes
    """

    def __init__(self, boxes_list):
        self.boxes_list = boxes_list

    def add(self, box):
        self.boxes_list.append(box)

    def merge(self, word_th=10, same_line_th=6):
        """
        Merge bounding boxes to construct phrases
        # TO DO TO REVIEW COMPLEXITY AND EXPLORE THE NATURE OF DATA
        :param word_th: (int) representing the number pixels separating two boxes
                              horizontally to consider them as closer
        :param same_line_th: (int) representing the pixels separating two boxes
                             horizontally to consider them in the same line
        """

        updated_list = True
        while updated_list:
            updated_list = False
            for i, curr_box in enumerate(self.boxes_list):
                for j in range(i + 1, len(self.boxes_list)):
                    next_box = self.boxes_list[j]
                    if not next_box.is_merged():
                        if (curr_box.is_closer_h(next_box, word_th, same_line_th)):
                            curr_box.merge(next_box)
                            next_box.set_merged()
                            del (self.boxes_list[j])
                            updated_list = True
                            break

    def to_array(self):
        """
        :return: (numpy array) of shape [number of boxes, 4, 2]
        """
        return np.array([b.to_array() for b in self.boxes_list])

    def save(self, image_path, img, dir_name, type="images"):
        """
        save bounding boxes
        :param image_path: (str) image path
        :param img:(numpy array) of shape [h, w, c]
        :param dir_name: (str) output folder path
        :param type: (str) 'images' or 'array' whether you want to save boxes in an image and see results or in
        a serialized numpy object file
        """
        if type == "images":
            file_utils.saveResult(image_path, img[:, :, ::-1], self.to_array(), dirname=dir_name)
        elif type == "array":
            with open("boxes.npy", "wb") as f:
                np.save(f, self.to_array())

    def display(self):
        for box in self.boxes_list:
            print(box)

    def mask(self, img):
        """
        a mothod that construct a binary image max black for background and white for bounding boxes areas
        :param img: (numpy array) of shape [h, w, c]
        :return: (numpy array) of shape [h, w, c]
        """
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for box in self.boxes_list:
            mask[box.p1[1]:box.p4[1], box.p1[0]:box.p2[0]] = 255
        return mask

    def detect_src_lang(self, translation_service):
        """
        :param translation_service: (str) "Google" or "TextBlob"
        :return: (str) encoding detected source language ex : "en", "fr",
        """
        src_language = None
        # detect source language if not mentioned
        all_text = " ".join([box.custom_text.content for box in self.boxes_list])
        if translation_service == "Google":
            src_language = Translator().detect(all_text).lang
        elif translation_service == "TextBlob":
            src_language = TextBlob(all_text).detect_language()

        return src_language

    def fill_translation(self,
                         empty_img,
                         same_block_th=10,
                         translation_service="Google",
                         src_language=None,
                         dest_language=None,
                         result_path=None):
        """
        :param empty_img: (numpy array) of shape [h, w, c] representing the inpainted image without text
        :param same_block_th:
        :param translation_service: (str) "Google" or "TextBlob"
        :param src_language: (str) encoding detected source language ex : "en", "fr",
        :param dest_language: (str) encoding detected destination language ex : "en", "fr",
        :param result_path: (str) final result path /home/.../res_img.png
        :return:
        """

        # refine closer lines font size (paragraphs) in a tricky way
        for box in self.boxes_list:
            for box2 in self.boxes_list:
                if box is not box2 and box.is_closer_v(box2, same_block_th=same_block_th):
                    box.refine_font_size(box2)

        # init image and draw
        img = Image.fromarray(empty_img.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        for box in self.boxes_list:
            translated_text = box.custom_text.content
            if translation_service == "Google":
                translator = Translator()
                translated_text = translator.translate(box.custom_text.content,
                                                       src=src_language, dest=dest_language).text
            elif translation_service == "TextBlob":
                try:
                    translated_text = str(TextBlob(box.custom_text.content). \
                                          translate(from_lang=src_language, to=dest_language))
                except exceptions.NotTranslated:
                    translated_text = box.custom_text.content

            # print(box.custom_text.content,"==>",translated_text,"=>",box.custom_text.color)
            # print("--------------------------------")
            font = ImageFont.truetype(box.custom_text.font, box.custom_text.font_size)
            draw.text((box.p1[0], box.p1[1]), r'{}'.format(translated_text),
                      box.custom_text.color, font=font)
        img.save(result_path)
