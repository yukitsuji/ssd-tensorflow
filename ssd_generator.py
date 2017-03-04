#!/usr/bin/env python
import sys
sys.path.append("./")

class Generator(obcjet):
    def __init__(self, images, labels, class_list, shuffle=True, imsize=(300, 300), color='RGB', mode='', callback=None):
        self._images = images
        self._data_size = len(images)
        self._labels = labels
        self._class_list = class_list
        self._shuffle = shuffle
        self._imsize = imsize
        self._color = color
        self._mode = mode
        self._callback = callback
        from load_for_detection.load_for_detection import make_nparray
        self._labels = make_nparray(self._labels, len(self._class_list))

    def batch(self, batch, callback=None):
        if self._shuffle:
            perm = np.random.permutation(self._data_size)
            batches = [perm[i * batch(i+1):(i + 1) * batch]
                      for i in range(int(np.ceil(self._data_size / batch)))]
        else:
            perm = np.arange(self._data_size)
            batches = [perm[i * batch(i+1):(i + 1) * batch]
                      for i in range(int(np.ceil(self._data_size / batch)))]
        imgfiles = [[self._images[p] for p in b] for b in batches]

        imgs = ImageLoader(imgfiles)

        import itertools
        from resize import resize

        for p, imgs in itertools.izip(batches, imgs.load()):
            labels = self._labels[p].copy()
            for index, img in enumerate(imgs):
                label = np.array([labels[index]], dtype=np.float32)
                imgs[index], labels[index] = resize(np.array(img, dtype=np.float32), size=self._imsize, labels=label, num_class=len(self._class_list))
            if self._callback is not None:
                imgs, labels = self._callback.create(np.array(imgs, dtype=np.float32), labels=labels, num_class=len(self._class_list))
            yield np.array(imgs, dtype=np.float32), labels

class ImageLoader(object):
    """load image pathes and its labels"""
    def __init__(self, image_pathes):
        """
           image_pathes(list): list of each batches
        """
        self._image_pathes = image_pathes

    def load(self):
        """
           imgs(list): list of numpy array
        """
        for batch_pathes in self._image_pathes:
            imgs = []
            for image_path in batch_pathes:
                img = self._readimg(image_path)
                imgs.append(img)
            yield imgs

    def _readimg(self, path):
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
        return img

def extract_label_from_xml(xml):
    """
       xml : TODO: define

       VOC pascal
       COCO
       udacity-self-driving
       others
    """
    pass
