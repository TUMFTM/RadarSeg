# Standard Libraries
import io

# 3rd Party Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def decode_figure(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it.

    Note: The supplied figure is closed and inaccessible after this call.

    Arguments:
        figure: Figure of a plot, <plt.figure>.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def get_colors(colour_scheme: str = 'primary'):
    """
    Provides an normalized numpy array of rgb colors according to TUM CI standard.

    Arguments:
        colour_scheme: Specifies the returned list of colors, <str>.

    Return:
        colors: Normalized array of rgb values (n, 3), <float>
    """
    # TUM CI primary colormap
    primary = \
        [[0, 101, 189],
         [0, 0, 0],
         [255, 255, 255]]

    # TUM CI secondary colormap
    secondary = \
        [[0, 82, 147],
         [0, 51, 89],
         [88, 88, 88],
         [156, 157, 159],
         [217, 217, 217]]

    # TUM CI accent colormap
    accent = \
        [[218, 215, 203],
         [227, 114, 34],
         [162, 173, 0],
         [152, 198, 234],
         [100, 160, 200]]

    # TUM CI extended colormap
    extended = \
        [[0, 0, 0],
         [105, 8, 90],
         [15, 27, 95],
         [0, 119, 138],
         [0, 124, 48],
         [103, 154, 29],
         [225, 220, 0],
         [249, 186, 0],
         [214, 76, 19],
         [196, 7, 27],
         [156, 13, 22]]

    # TUM blues
    blues = \
        [[0, 0, 0],
         [0, 101, 189],
         [100, 160, 200],
         [152, 198, 234]]

    colors = {'primary': primary, 'secondary': secondary, 'accent': accent, 'extended': extended, 'blues': blues}
    return np.array(colors[colour_scheme], np.float) / 255


def rgb2gray(rgb):
    """
    Converts rgb values to grayscale intensity.

    Arguments:
        rgb: List of rgb values, <List>.

    Returns:
        Grayscale intensity.
    """
    return 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
