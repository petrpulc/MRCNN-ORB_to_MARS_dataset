import colorsys
import os
import random

import matplotlib as plt
import numpy as np
from cv2 import cvtColor, COLOR_GRAY2RGB
from matplotlib.pyplot import subplots

from motion.object import Object

ax = None


def generate_colors(n, randomize=True):
    """
    Generate random, visually distinct colors.

    :param n: number of colors to generate
    :param randomize: shuffle the resulting list of colors
    :return: array of n colors with different hue
    """
    hsv = [(i / n, 1, 1) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if randomize:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Overlay part of image with color.

    :param image: original image
    :param mask: mask to be coloured
    :param color: color to paint over the image
    :param alpha: translucency of the paint
    :return: image with overlay
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def plot_image(image, tracker, path=''):
    global ax
    if ax is None:
        height, width = image.shape[:2]
        _, ax = subplots(1, figsize=(16, 16))
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
    ax.cla()
    ax.axis('off')

    masked_image = cvtColor(image, COLOR_GRAY2RGB)
    ax.imshow(masked_image)
    for obj in tracker.objects:  # type: Object
        y1, x1, y2, x2 = obj.roi
        if not obj.roi_valid:
            continue

        p = plt.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.5, linestyle="dashed",
                                  edgecolor=obj.color, facecolor='none')
        ax.add_patch(p)
        caption = "{}".format(obj.id)
        t = ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")
        t.set_bbox({'facecolor': obj.color, 'alpha': 0.2, 'pad': 0, 'edgecolor': obj.color})

    #plt.pyplot.savefig(os.path.join(path, 'rois', '{:06}.pdf'.format(tracker.frame_no)))

    for obj in tracker.objects:
        for point in obj.points:
            for i in (x for x in point.history if x > tracker.frame_no - 15):
                a = 1 / (tracker.frame_no - i)
                if i + 1 in point.history:
                    ax.arrow(*point.history[i], *(point.history[i + 1] - point.history[i]), head_width=2,
                             color=obj.color, alpha=a)
                # else:
                #    ax.plot(*point.history[i], marker='o', markersize=1, color=obj.color, alpha=a)

    # for point in tracker.loose_points:
    #     for i in (x for x in point.history if x > tracker.frame_no - 5):
    #         a = 1 / (tracker.frame_no - i)
    #         if i + 1 in point.history:
    #             ax.arrow(*point.history[i], *(point.history[i + 1] - point.history[i]), head_width=2,
    #                      color='gray', alpha=a)

    plt.pyplot.savefig(os.path.join(path, 'full', '{:06}.pdf'.format(tracker.frame_no)))


def plot_mrcnn(image, r, tracker, path=''):
    global ax
    if ax is None:
        height, width = image.shape[:2]
        _, ax = subplots(1, figsize=(16, 16))
        ax.set_ylim(height, 0)
        ax.set_xlim(0, width)
    ax.cla()
    ax.axis('off')

    masked_image = cvtColor(image, COLOR_GRAY2RGB)

    N = len(r['scores'])
    colors = generate_colors(N)

    for i in range(N):  # type: Object
        color = colors[i]
        y1, x1, y2, x2 = r['rois'][i]
        p = plt.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.5, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
        ax.add_patch(p)

        caption = "{} {:.2f}".format(r['class_ids'][i], r['scores'][i])
        t = ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")
        t.set_bbox({'facecolor': color, 'alpha': 0.2, 'pad': 0, 'edgecolor': color})

    ax.imshow(masked_image)
    plt.pyplot.savefig(os.path.join(path, 'mrcnn', '{:06}.pdf'.format(tracker.frame_no)))
