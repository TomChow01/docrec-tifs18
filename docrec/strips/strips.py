import os
import re
import numpy as np
import cv2
from random import shuffle
import matplotlib.pyplot as plt

from .strip import Strip


class Strips(object):
    ''' Strips operations manager.'''

    def __init__(
        self, path=None, obj=None, filter_blanks=True, blank_tresh=127
    ):
        ''' Strips constructor.

        @path: path to strips (in case of load real strips)
        '''

        assert (path is not None) or (obj is not None)

        self.strips = []
        if path is not None:
            assert os.path.exists(path)
            self._load_data(path)
        else:
            self.strips = map(lambda strip: strip.copy, obj.strips)

        if filter_blanks:

            self.strips = [strip for strip in self.strips if not strip.is_blank(blank_tresh)]


    def __call__(self, i):
        ''' Returns the i-th strip. '''

        return self.strips[i]


    def _load_data(self, path, regex_str='.*\d\d\d\d\d\.*'):
        ''' Stack strips horizontally.

        Strips are images with same basename (and extension) placed in a common
        directory. Example:

        basename="D001" and extension=".jpg" => strips D00101.jpg, ..., D00130.jpg.
        '''

        path_images = '{}/strips'.format(path)
        path_masks = '{}/masks'.format(path)
        regex = re.compile(regex_str)

        # loading images
        fnames = sorted([fname for fname in  os.listdir(path_images) if regex.match(fname)])
        images = []
        for fname in fnames:
            image = cv2.cvtColor(
                cv2.imread('{}/{}'.format(path_images, fname)),
                cv2.COLOR_BGR2RGB
            )
            images.append(image)

        # load masks
        masks = []
        if os.path.exists(path_masks):
            for fname in fnames:
                mask = np.load('{}/{}.npy'.format(path_masks, os.path.splitext(fname)[0]))
                masks.append(mask)
        else:
            masks = len(images) * [None]
            self.artificial_mask = True

        for position, (image, mask) in enumerate(zip(images, masks), 1):
            strip = Strip(image, position, mask)
            self.strips.append(strip)


#
#    def copy(self):
#        ''' Return a copy of object. '''
#
#        strips = self.__class__()
#        strips.__dict__ = self.__dict__.copy()
#
#        return strips
#
#
#    def dump(self, path):
#        ''' Store fields into the folder pointed by path. '''
#
#        path = os.path.join(path, 'strips')
#        if not os.path.exists(path):
#            os.makedirs(os.path.join(path, 'images'))
#
#        for i, image in enumerate(self.images):
#            np.save(os.path.join(path, 'images', '%s.npy' % i), image)
#
#        with open(os.path.join(path, 'order.txt'), 'w') as f:
#            f.write(' '.join([str(x) for x in self.order]))
#
#
    def trim(self, left=0, right=0):
        ''' Trim borders from strips. '''

        n = len(self.strips)
        self.strips = self.strips[left : n - right]
        return self


    def image(self, order=None):
        ''' Return image in a specific order. '''

        strips = self.strips
        if order is not None:
            strips = [self.strips[i] for i in order]
        image = strips[0].copy()
        for strip in strips[1 :]:
            image.stack(strip)
        return image.image

##
##   # Returns a subset containing the n first strips
##   def firstn(self, n=30):
##      self.shuffle(range(n))
##      return self
##
##   def complexity(self):
##      strips_bin = self.copy().conv2bin()
##      h, w = strips_bin.image().shape
##      norm_factor = h * (w - 1)
##      return float(np.diff(strips_bin.image(), axis=1).sum()) / norm_factor
##
##   def save_image(self, filename):
##      imsave(filename, self._img)
##
##    def direct_comparison(self, order):
##        ''' Accuracy by neighbor comparsison. '''
##
##        assert len(order) > 0
##        assert len(self.order) > 0
##
##        new_order = np.array([self.order[p] for p in order])
##        ref_order = np.arange(len(new_order))
##
##        return (new_order == ref_order).sum() / float(ref_order.size)
##

    def plot(self, size=(8, 8), fontsize=6, ax=None, show_lines=False):
        ''' Plot strips given the current order. '''

        assert len(self.strips) > 0
        if ax is None:
            fig = plt.figure(figsize=size, dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig = None

        shapes = [[strip.h, strip.w] for strip in self.strips]
        max_h, max_w = np.max(shapes, axis=0)
        sum_h, sum_w = np.sum(shapes, axis=0)

        # Background
        offsets = [0]
        background = self.strips[0].copy()
        for strip in self.strips[1 :]:
            offset = background.stack(strip)
            offsets.append(offset)

        ax.imshow(background.image)
        ax.axis('off')

        for strip, offset in zip(self.strips, offsets):
            d = strip.w / 2
            ax.text(
                offset + d, 50, str(strip.position), color='blue',
                fontsize=fontsize, horizontalalignment='center'
            )

        if show_lines:
            ax.vlines(
                offsets[1 :], 0, max_h, linestyles='dotted', color='red',
                linewidth=0.5
            )

        return fig, ax, offsets
