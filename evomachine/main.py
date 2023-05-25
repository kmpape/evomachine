# simple single image acquisition example with snap
from pycromanager import Core
import numpy as np
import matplotlib.pyplot as plt
import time

#### Setup ####
core = Core()

#### imaging settings
exposure = 20
num_frames_to_capture = 100
core.set_exposure(exposure)

#### strobe settings
# by enforcing a strobe (a small delay between acquisitions), our snap_image acquisition framerate becomes an order of magnitude more precise (as of 20201006)
interframe_interval = 50
assert interframe_interval > exposure

#### holder variables for images, model values, and processing timestamps
frames = []
model = []
acq_timestamps = []
process_timestamps = []

import numpy as np
import cv2
from numba import jit


# image processing convenience class
# you might have to install some other python dependencies, but this should be clear while trying to run demo
class ImageProcessor:
    def __init__(self, ops=None):

        if ops is None:
            ops = build_ops()

        # store ops
        self.ops = ops

        # helper fxn to initialize a template filter for
        self.template_filter = gaussian_2d(
            1 + 6 * self.ops["template_filter_width"], self.ops["template_filter_width"]
        )

    def set_ops(self, ops):
        """ helper wrapper to change imageprocessor params, to be called if you want to update default parameters for image processing after object instantiation """

        self.ops = ops

    def segmentchunk(self, frame):
        """ wrapper function to take a frame, segment local peaks, and merge adjacent peaks """

        # image preprocessing
        filt_frame = image_filtering(
            frame,
            self.ops["fb_post_threshold"],
            self.ops["fb_threshold_margin"],
            self.ops["med_filt_size" ""],
            self.template_filter,
        )

        # find centers of blobs
        xs, ys = find_centers(filt_frame)

        if len(xs) == 0:
            return []

        # merge centers too close to one another
        xs, ys = merge_centers(xs, ys, self.ops["fb_min_blob_spacing"])

        return list(zip(xs, ys))


def image_filtering(
        frame,
        fb_post_threshold=50,
        fb_threshold_margin=50,
        med_filt_size=5,
        template_filter=None,
):
    """
    :param frame: numpy array (image)
    :param fb_post_threshold: threshold to apply after template matching
    :param fb_threshold_margin: threshold to apply before template matching (arg comes after fb_post_threshold for legacy)
    :param med_filt_size: size of median filter kernel
    :param template_filter: can take prespecified template for filtering (numpy array)
    :return: numpy array (image)
    """

    # apply threshold
    threshold = np.median(frame) + fb_threshold_margin
    frame = (frame > threshold) * frame

    # apply median filter
    frame = cv2.medianBlur(frame, med_filt_size)

    # template filtering
    frame = cv2.matchTemplate(frame, template_filter, cv2.TM_CCOEFF)

    # pad frame b/c of template matching
    p = (len(template_filter) // 2) + 1
    frame = np.pad(frame, (p, 0), "constant")

    return (frame > fb_post_threshold) * frame


@jit(nopython=True)
def find_centers(frame):
    """
    function navigate up gradient to identify local peaks
    :param frame: numpy array
    :return: (xs, ys) lists of blob centers, indexed by blob number
    """

    # skipping edge pixels
    sd = frame.shape
    edg = 3
    [x, y] = np.where(frame[edg: sd[0] - edg, edg: sd[1] - edg - 1])

    x = x + edg - 1
    y = y + edg - 1

    # intialize lists with item (necessary for legacy numba build, should be able to remove depending on numba version)
    xs = [0]
    ys = [0]

    for j in range(0, len(y)):
        if (
                (frame[x[j], y[j]] >= frame[x[j] - 1, y[j] - 1])
                and (frame[x[j], y[j]] >= frame[x[j] - 1, y[j]])
                and (frame[x[j], y[j]] >= frame[x[j] - 1, y[j] + 1])
                and (frame[x[j], y[j]] >= frame[x[j], y[j] - 1])
                and (frame[x[j], y[j]] >= frame[x[j], y[j] + 1])
                and (frame[x[j], y[j]] >= frame[x[j] + 1, y[j] - 1])
                and (frame[x[j], y[j]] >= frame[x[j] + 1, y[j]])
                and (frame[x[j], y[j]] >= frame[x[j] + 1, y[j] + 1])
        ):

            # ridge/mesa consolidation code
            # find horizontal neighbor pixels of equal value
            if frame[x[j], y[j]] == frame[x[j] - 1, y[j]]:
                pass

            # find horizontal neighbor pixels of equal value
            elif frame[x[j], y[j]] == frame[x[j], y[j] - 1]:
                pass
            else:
                xs.append(x[j])
                ys.append(y[j])

    return xs[1:], ys[1:]


@jit(nopython=True)
def merge_centers(xs, ys, fb_min_blob_spacing):
    """
    function to merge adjacent detected peaks
    :param xs: list of x locations of blobs, indexed by blob number
    :param ys: list of y locations of blobs, indexed by blob number
    :param fb_min_blob_spacing: minimum distance within which, adjacent blobs are merged
    :return: (xs, ys) lists of blob center coords
    """

    # grab number of centers
    numcenters = len(xs)

    # code for removing rois
    code = 9999

    if numcenters != 0:

        # calculate distance between all centers
        # set all values to dummy high values and greedy alg search for min
        dist_matrix = np.ones((numcenters, numcenters)) * 200

        # calculate distances
        for n in range(numcenters):
            for n2 in range(n + 1, numcenters):
                # calculate distance
                dist = np.sqrt(((xs[n] - xs[n2]) ** 2) + ((ys[n] - ys[n2]) ** 2))

                # add to distance matrix
                dist_matrix[n, n2] = dist

        # greedy alg remove minimum distance till we're small enough
        mindist = np.amin(dist_matrix)
        while mindist < fb_min_blob_spacing:
            # get min distance and min pixel pair
            mindist = np.amin(dist_matrix)
            minpixelpair = np.argmin(dist_matrix)

            # fit linear ndx to larger ndx
            c2 = minpixelpair % numcenters

            # kill c2
            xs[c2] = code
            ys[c2] = code

            # remove acccording to code in matrix
            dist_matrix[:, c2] = code
            dist_matrix[c2, :] = code

    # remove entries with code
    xs = [x for x in xs if x != code]
    ys = [y for y in ys if y != code]

    return xs, ys


@jit(nopython=True)
def gaussian_2d(im_width, sigma):
    """
    function to generate a gaussian blob, centered at middle of numpy array
    :param im_width: width of frame
    :param sigma: param for gaussian
    :return: im_width x im_width numpy array
    """

    g = np.zeros((im_width, im_width), dtype=np.float32)

    # gaussian filter
    for i in range(int(-(im_width - 1) / 2), int((im_width + 1) / 2)):
        for j in range(int(-(im_width - 1) / 2), int((im_width + 1) / 2)):
            x0 = int((im_width) / 2)  # center
            y0 = int((im_width) / 2)  # center
            x = i + x0  # row
            y = j + y0  # col
            g[y, x] = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2 / sigma / sigma)

    return g


def build_ops():
    """
    function to pack default parameters for segmentation into a dict
    """

    # quant params
    fb_threshold_margin = 40
    fb_min_blob_spacing = 8
    fb_post_threshold = 40
    cb_maxdist = 9
    template_filter_width = 3
    med_filt_size = 5

    ops = {
        "fb_threshold_margin": fb_threshold_margin,
        "fb_min_blob_spacing": fb_min_blob_spacing,
        "fb_post_threshold": fb_post_threshold,
        "cd_maxdist": cb_maxdist,
        "template_filter_width": template_filter_width,
        "med_filt_size": med_filt_size,
    }

    return ops


# in this case we're doing some image processing, finding local peaks, and taking a 3x3 grid of pixel values from each peak
# this function returns an average across all detected peaks
def process_frame(frame, ip, is_demo=False):
    # if we're running this example with the micromanager demo config, peakfinding doesn't really make sense on gratings
    if is_demo:
        return 0

    # simple peakfinding algorithm from accompanying module
    xys_list = ip.segmentchunk(frame.astype(np.float32))

    # if no peaks, return placeholder value
    if len(xys_list) == 0:
        return 0

    # grab 3x3 pixels around each peak
    pix = []
    xys = np.array(xys_list) - 1  # -1 because of single pixel offset bug...
    pix.append(frame[xys[:, 0], xys[:, 1]])
    pix.append(frame[xys[:, 0], xys[:, 1] - 1])
    pix.append(frame[xys[:, 0], xys[:, 1] + 1])
    pix.append(frame[xys[:, 0] - 1, xys[:, 1]])
    pix.append(frame[xys[:, 0] - 1, xys[:, 1] - 1])
    pix.append(frame[xys[:, 0] - 1, xys[:, 1] + 1])
    pix.append(frame[xys[:, 0] + 1, xys[:, 1]])
    pix.append(frame[xys[:, 0] + 1, xys[:, 1] - 1])
    pix.append(frame[xys[:, 0] + 1, xys[:, 1] + 1])

    # flatten and sort peak-averages
    peak_averages = np.sort(np.array(pix).mean(axis=0).flatten())

    # in this example let's just average across peaks
    avg = peak_averages.mean()

    return avg


#### quantification settings
# initialize jit precompilation with an intial image from the microscope
ip = ImageProcessor()
core.snap_image()
tagged_image = core.get_tagged_image()
frame = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
garbage = process_frame(frame, ip)


# define a function that triggers your perturbation/event based on quantified measurements
# in this demo we're extracting average pixel intensities from blobs within image. thus the "model" can simply
# be a list of values, and our process model function simply detects a rise over the threshold of the latest measurement
# however our pycro-manager demo camera returns an image of sinusoidal gratings so peakfinding returns nonsense
# so, here is just a placeholder function (it's over 9000 lol)
def process_model(model, threshold=9000):
    if model[-1] > threshold:
        # code here to do whatever perturbation you want
        pass

    return


#### do acquisition
print('beginning acquisition...')
t0 = time.time()
next_call = time.time()  # updated periodically for when to take next image
for f in range(num_frames_to_capture):

    # snap image
    core.snap_image()
    tagged_image = core.get_tagged_image()

    # save acquisition time timestamp
    t1 = time.time()
    acq_timestamps.append(time.time() - t0)

    # pixels by default come out as a 1D array. We can reshape them into an image
    frame = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])

    # quantify image and save processing time timestamp
    val = process_frame(frame, ip, is_demo=True)
    process_timestamps.append(time.time() - t1)

    # store latest value in model and conditionally trigger perturbation
    model.append(val)
    process_model(model)

    # helpful printout to monitor progress
    if f % 50 == 0:
        print('current frame: {}'.format(f))

    # wait until we're ready to snap next image. note that in this example, the first few images may exceed the strobe delay as numba jit compiles the relevant python functions
    nowtime = time.time()
    next_call = next_call + interframe_interval / 1000
    if next_call - nowtime < 0:
        print("warning: strobe delay exceeded inter-frame-interval on frame {}.".format(f))
    else:
        time.sleep(next_call - nowtime)

print('done!')
