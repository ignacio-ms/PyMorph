import numpy as np
from scipy.optimize import curve_fit


def compute_z_profile(image_stack, mask_stack):
    """
    Compute the mean intensity profile along the z-axis of an image stack.
    Using a segmentation mask to separate foreground and background.
    :param image_stack:
    :param mask_stack:
    :return: Foreground and background mean intensities at each z-slice.
    """
    z_slices = image_stack.shape[2]

    # Binary mask of the nuclei
    mask_stack = mask_stack > 0

    fg_means = []
    bg_means = []

    for z in range(z_slices):
        slice_2d = image_stack[..., z]
        mask_2d = mask_stack[..., z]

        try:
            # Foreground (nuclei) intensities
            fg_pixels = slice_2d[mask_2d]

            # Background intensities
            bg_pixels = slice_2d[~mask_2d]

            fg_means.append(np.mean(fg_pixels))
            bg_means.append(np.mean(bg_pixels))
        except Exception as e:
            print(f"Error at z={z}: {e}")
            fg_means.append(0)
            bg_means.append(0)

    return np.array(fg_means), np.array(bg_means)


def compute_z_profile_no_mask(image_stack):
    """
    Compute the mean intensity profile along the z-axis of an image stack.
    The foreground mask will be assessed by a simple min between max umbralization.
    :param image_stack:
    :return: Foreground and background mean intensities at each z-slice.
    """
    z_slices = image_stack.shape[2]
    fg_means = []
    bg_means = []

    for z in range(z_slices):
        slice_2d = image_stack[..., z]
        umbral = np.min([np.max(slice_2d), np.mean(slice_2d) + 3 * np.std(slice_2d)])
        mask_2d = slice_2d > umbral

        try:
            fg_pixels = slice_2d[mask_2d]
            bg_pixels = slice_2d[~mask_2d]

            fg_means.append(np.mean(fg_pixels))
            bg_means.append(np.mean(bg_pixels))
        except Exception as e:
            print(f"Error at z={z}: {e}")
            fg_means.append(0)
            bg_means.append(0)

    return np.array(fg_means), np.array(bg_means)


def logistic_inverted(x, L, U, k, x0):
    """
    Inverted logistic function that goes from ~U down to ~L:
    f(x) = L + (U - L) / [1 + exp(k * (x - x0))]
    """
    return L + (U - L) / (1.0 + np.exp(k * (x - x0)))


def fit_inverted_logistic(xdata, ydata, p0=None, maxfev=50000):
    """
    Fit the inverted logistic model to (xdata, ydata).
    p0 = (L, U, k, x0) if you want to supply guesses manually.
    :param xdata: X data.
    :param ydata: Y data.
    :param p0: Initial guess for the parameters. (Default: None)
    :param maxfev: Maximum number of function evaluations. (Default: 50000)
    """
    mask = np.isfinite(xdata) & np.isfinite(ydata)
    x_valid, y_valid = xdata[mask], ydata[mask]

    if len(x_valid) < 5:
        raise ValueError("Not enough data points to fit a logistic.")

    # If no initial guess is provided, estimate from data
    if p0 is None:
        L_init = np.min(y_valid)
        U_init = np.max(y_valid)
        # x0 ~ the midpoint of x range (rough guess)
        x0_init = 0.5 * (x_valid.min() + x_valid.max())
        # slope guess k from the 'steepest' region; let's just pick something small
        k_init = 0.01
        p0 = [L_init, U_init, k_init, x0_init]

    popt, pcov = curve_fit(
        logistic_inverted,
        x_valid,
        y_valid,
        p0=p0,
        maxfev=maxfev
    )
    return popt


def correction_factor(z, popt, z_ref=0.0):
    """
    Compute CF(z) = f(z_ref)/f(z),
    where f is the inverted logistic model.
    popt = [L, U, k, x0] from the curve fit.
    :param z: Z-coordinate.
    :param popt: Fitted parameters.
    :param z_ref: Reference Z-coordinate.
    """
    L, U, k, x0 = popt
    f_z = logistic_inverted(z, L, U, k, x0)
    f_ref = logistic_inverted(z_ref, L, U, k, x0)  # reference slice intensity
    # Avoid divide-by-zero if any f_z is ~ 0
    return np.where(f_z != 0, f_ref / f_z, 1.0)


def plot_calibration(z_indices, fg_means, bg_means, popt, title=None):
    """
    Plot the calibration curve and the intensity profiles.
    :param z_indices: Z indices.
    :param fg_means: Foreground mean intensities.
    :param bg_means: Background mean intensities.
    :param popt: Fitted parameters.
    :param title: Plot title.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    plt.plot(z_indices, fg_means, 'o-', label='Foreground Means')
    plt.plot(z_indices, bg_means, 'o-', label='Background Means')

    # Plot the fitted curve
    x_fit = np.linspace(z_indices.min(), z_indices.max(), len(z_indices))
    y_fit = logistic_inverted(x_fit, *popt)
    plt.plot(x_fit, y_fit, '--', label='Fitted Decay Curve')

    plt.xlabel('Z-Slice Index')
    plt.ylabel('Mean Intensity')
    plt.title(title if title else 'Z-Profile Calibration')
    plt.legend()
    plt.grid(True)
