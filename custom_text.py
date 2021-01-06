class CustomText:
    """
    A class representing a text in a custom way
    TO DO : complete and improve in next version
    """
    def __init__(self, content, font_file, font_size, color, italic=False,
                 bold=False, underlined=False):
        """
        :param content: (str)
        :param font_file: (str) file path to the font file
        :param font_size: (number : float or int or ...) bounding box height to estimate the font size
        :param color: (tuple) of shape (3,) representing the color (R, G, B)
        :param italic: (Boolean)
        :param bold: (Boolean)
        :param underlined: (Boolean)
        """
        self.content = content
        # convert pixels to pt 1px = 0.75pt we make it to 0.95 experimentally
        self.font_size = int(font_size * 0.95)
        self.font = font_file
        self.color = color
        self.italic = italic
        self.bold = bold
        self.underlined = underlined

    def append(self, is_precedent, sep, custom_text2):
        """
        concatenate two custom objects
        :param is_precedent: (Boolean) if the self is preceding custom_text2 to fix the order of concatenation
        :param sep: (str) the separator
        :param custom_text2: (custom_text object)
        """
        if is_precedent:
            self.content = self.content + sep + custom_text2.content
        else:
            self.content = custom_text2.content + sep + self.content

        # font_size
        self.font_size = self.avg_font_size(custom_text2)

    def avg_font_size(self, custom_text2):
        """
        :param custom_text2: (custom_text object)
        :return: average font size between self and custom_text2
        """
        return int((self.font_size+custom_text2.font_size)/2)

    def refine_font_size(self, custom_text2):
        """
        set self font size to avg(self,other) generally used when concatenating two custom objects
        :param custom_text2: (custom_text object)
        """
        self.font_size = self.avg_font_size(custom_text2)
        custom_text2.font_size = self.font_size