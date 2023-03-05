import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from torch import from_numpy, FloatTensor
from torchvision import transforms


# ============== general functions ===========
def get_mean_std_of_cube_data(img_dataset_mean, img_dataset_std, img_len, background_len, background_value):
    mean_s1 = img_dataset_mean
    std_s1 = img_dataset_std
    len_s1 = img_len
    mean_s2 = background_value
    std_s2 = 0.
    len_s2 = background_len
    mean_union = (mean_s1 * len_s1 + mean_s2 * len_s2) / (len_s1 + len_s2)
    # Compute the standard deviation of the union
    std_union = ((std_s1 ** 2 + mean_s1 ** 2) * len_s1 + (std_s2 ** 2 + mean_s2 ** 2) * len_s2
                 - mean_union ** 2 * (len_s1 + len_s2)) / (len_s1 + len_s2)
    std_union = np.sqrt(std_union)
    return mean_union, std_union

# ============== sphere functions ============


def lift_point(x, y, z, A, B, C, R):
    """ take point p=(x, y, z) sphere with center (A,B,C) with radius R, and north pole q=(A,B,C+R).
        finds intersection of the line pq with sphere"""
    x_coord = A + (2 * R * (x - A) * (C + R - z) / ((x - A) ** 2 + (y - B) ** 2 + (-z + C + R) ** 2))
    y_coord = B + (2 * R * (y - B) * (C + R - z) / ((x - A) ** 2 + (y - B) ** 2 + (-z + C + R) ** 2))
    z_coord = C + R - (2 * R * ((C + R - z) ** 2) / ((x - A) ** 2 + (y - B) ** 2 + (-z + C + R) ** 2))
    return np.array([x_coord, y_coord, z_coord])


def lift_image(image, A, B, C, R):
    xs = np.arange(0, image.shape[0], 1)
    ys = np.arange(0, image.shape[1], 1)
    points = np.array([lift_point(x, y, 0, A, B, C, R) for x in xs for y in ys])
    return points.reshape((image.shape[0], image.shape[1], 3))


def naive_discretize_to_lattice(img, lifted_points, depth_factor, background_value=0):
    xs = np.arange(0, img.shape[0], 1)
    ys = np.arange(0, img.shape[1], 1)
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    d = max(h, w) * depth_factor
    valid = min(h, w, d) - 1
    cube = np.ones((h, w, d, c)) * background_value
    for i in xs:
        for j in ys:
            point = lifted_points[i, j]
            whole_point = np.array([max(min(round(coord), valid), 0) for coord in point])
            if not np.any(cube[whole_point[0], whole_point[1], whole_point[2]]):
                cube[whole_point[0], whole_point[1], whole_point[2]] = img[i, j]
    return cube


def naive_discretize_with_blurr(img, lifted_points, depth_factor, background_value=0):
    cube = naive_discretize_to_lattice(img, lifted_points, depth_factor, background_value)
    smoothing = GaussianSmoothing(1, 5, 1, dim=3)
    tensor_cube = from_numpy(cube).type(FloatTensor).reshape(1, 1, cube.shape[0], cube.shape[1], cube.shape[2])
    tensor_cube = F.pad(tensor_cube, (2, 2, 2, 2, 2, 2))
    smooth_cube = smoothing(tensor_cube)
    return np.array(smooth_cube).reshape(cube.shape)


def spread_discretize(img, lifted_points, depth_factor, background_value=0):
    # angels of all points
    # new angels
    # new lifted points
    spread_lifted_points = {1:2}
    cube = naive_discretize_to_lattice(img, spread_lifted_points, depth_factor, background_value=0)
    return cube


def project_point(a, b, c, A, B, C, R):
    x_coord = (a - A) * (C + R) / (C + R - c) + A
    y_coord = (b - B) * (C + R) / (C + R - c) + B
    return np.array([x_coord, y_coord])


def get_so3_mat(phi, theta, psi):
    cos = np.cos
    sin = np.sin
    mat = np.array([[cos(phi) * cos(psi) - cos(theta) * sin(phi) * sin(psi),
                     -cos(phi) * sin(psi) - cos(theta) * sin(phi) * cos(psi), sin(phi) * sin(theta)],
                    [sin(phi) * cos(psi) + cos(theta) * cos(phi) * sin(psi),
                     -sin(phi) * sin(psi) + cos(theta) * cos(phi) * cos(psi), -cos(phi) * sin(theta)],
                    [sin(psi) * sin(theta), cos(psi) * sin(theta), cos(theta)]])
    return mat


def rotate_image_points_on_sphere(img_points, A, B, C, phi, theta, psi):
    rotated_mat = get_so3_mat(phi, theta, psi)
    points = np.array([rotated_mat.dot(np.array([img_points[i, j][0]-A, img_points[i, j][1]-B, img_points[i, j][2]-C])) for i in range(img_points.shape[0]) for j in range(img_points.shape[1])])
    points = np.array([[p[0]+A, p[1]+B, p[2]+C] for p in points])
    return points.reshape((img_points.shape[0], img_points.shape[1], 3))


# ==================== mobius functions ===================

def sphere_mobius(point, A, B, C, R, phi, theta, psi):
    # lift point from plane to sphere
    sphere_point = lift_point(point[0], point[1], 0, A, B, C, R)
    # apply SO(3) element on point
    centered_point = np.array([sphere_point[0]-A, sphere_point[1]-B, sphere_point[2]-C])
    rotated_mat = get_so3_mat(phi, theta, psi)
    rotated_point = rotated_mat.dot(centered_point)
    un_centered_point = np.array([rotated_point[0]+A, rotated_point[1]+B, rotated_point[2]+C])
    # project point from sphere to plane
    projected_point = project_point(un_centered_point[0], un_centered_point[1], un_centered_point[2], A, B, C, R)
    return projected_point


def mobius_trans(point, a, b, c, d):
    z = complex(point[0], point[1])
    w = (a * z + b) / (c * z + d)
    return np.array([np.real(w), np.imag(w)])


def apply_mobius_by_sphere_rot(img, A, B, C, R, phi, theta, psi):
    """

    :param img: image - (H,W,C), dtype=uint8[0...255]
    :param A: x coord of sphere center
    :param B: y coord of sphere center
    :param C: z coord of sphere ceAnter
    :param R: sphere radius
    :param phi: rotation around z axis
    :param theta: rotation around new x axis
    :param psi: rotation around new new z axis
    :return: transformed image
    """
    height = img.shape[0]
    width = img.shape[1]
    complex_zeros = [complex(0, 0)] * height * width  # zero
    transformed_centered_grid = np.array(complex_zeros).reshape(height, width)  # Gaussian integers (lattice points)
    for i in range(0, height):
        for j in range(0, width):
            trans_point = sphere_mobius(np.array([i - height // 2, j - width // 2]), A - height // 2, B - width // 2, C, R, phi, theta, psi)
            transformed_centered_grid[i, j] = complex(trans_point[0], trans_point[1])
    rows = np.array(list(range(0, height)) * width).reshape(width, height).T  # rows index
    columns = np.array(list(range(0, width)) * height).reshape(height, width)  # columns index

    # out_img = np.ones(img.shape, dtype=np.uint8) * 255 * 0  # zero image
    out_img = np.ones(img.shape, dtype=np.float32) * 255 * 0

    first = np.real(transformed_centered_grid) * 1
    second = np.imag(transformed_centered_grid) * 1
    first = first.astype(int)
    second = second.astype(int)
    f1 = first >= - height // 2
    f2 = first < height // 2
    f = f1 & f2
    s1 = second >= - width // 2
    s2 = second < width // 2
    s = s1 & s2
    combined = s & f
    # out_img - transformed image without interpolation
    out_img[first[combined] + height // 2, second[combined] + width // 2, :] = img[rows[combined], columns[combined], :]

    out_img_interpolated = out_img.copy()
    u = [True] * height * width
    canvas = np.array(u).reshape(height, width)
    canvas[first[combined] + height // 2, second[combined] + width // 2] = False
    converted_empty_index = np.where(canvas == True)
    converted_first = converted_empty_index[0]
    converted_second = converted_empty_index[1]

    new = converted_first.astype(complex) - height // 2
    new.imag = converted_second - width // 2

    ori = [sphere_mobius(np.array([p.real, p.imag]), A - height // 2, B - width // 2, C, R, -psi,
                   -theta, -phi) for p in new]
    ori = np.array([complex(p[0] + height//2, p[1] + width//2) for p in ori], dtype=np.complex128)
    p = np.hstack([ori.real, ori.real, ori.real])
    k = np.hstack([ori.imag, ori.imag, ori.imag])
    zero = np.zeros_like(ori.real)
    one = np.ones_like(ori.real)
    two = np.ones_like(ori.real) * 2
    third = np.hstack([zero, one, two])
    number_of_interpolated_point = len(one)
    e = number_of_interpolated_point
    interpolated_value_unfinished = map_coordinates(img, [p, k, third], order=1, mode='constant', cval=0)
    t = interpolated_value_unfinished

    interpolated_value = np.stack([t[0:e], t[e:2 * e], t[2 * e:]]).T
    if img.shape[2] == 1:   # grayscale
        out_img_interpolated[converted_first, converted_second, :] = interpolated_value[:, 0].reshape(-1, 1)
    else:
        out_img_interpolated[converted_first, converted_second, :] = interpolated_value
    return out_img, out_img_interpolated


# ============ plot functions ===========

def plot_lifted_points(points, image, A, B, C, R, cmap=None):
    xs_proj, ys_proj, zs_proj = points[:, :, 0], points[:, :, 1], points[:, :, 2]
    colors = image.reshape(image.shape[0]*image.shape[1]) if len(image.shape) == 2 else image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(A-R, A+R)
    ax.set_ylim(B-R, B+R)
    ax.set_zlim(C-R, C+R)
    if cmap:
        ax.scatter(xs_proj, ys_proj, zs_proj, c=colors, cmap=cmap)
    else:
        ax.scatter(xs_proj, ys_proj, zs_proj, c=colors)
    plt.show()


def plot_cube(cube, color_condition, indices_condition, cmap=None):
    """ assumes H x W x x D x C. If tensor might need to transpose: np.transpose(cube, (1, 2, 3, 0)) """
    # TODO: add support for RGB
    indices = np.array(np.meshgrid(range(cube.shape[0]), range(cube.shape[1]), range(cube.shape[2]), indexing='ij')).transpose((1, 2, 3, 0))
    i_xs, i_ys, i_zs = np.where(indices_condition(indices))
    c_xs, c_ys, c_zs = np.where(np.any(color_condition(cube), axis=-1))
    # TODO: intersect triplets
    i_p = np.stack([i_xs, i_ys, i_zs], axis=-1)
    c_p = np.stack([c_xs, c_ys, c_zs], axis=-1)
    intersections = set(map(tuple, i_p)).intersection(set(map(tuple, c_p)))
    intersections = np.array(list(intersections))
    xs, ys, zs = intersections[:, 0], intersections[:, 1], intersections[:, 2]
    colors = cube[xs, ys, zs]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 28)
    ax.set_zlim(0, 28)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if cmap:
        ax.scatter(xs, ys, zs, c=colors, cmap=cmap)
    else:
        ax.scatter(xs, ys, zs, c=colors)
    plt.show()


# ============== transforms ===============


class ToSphereTransform(object):
    def __init__(self, sphere_x_factor, sphere_y_factor, sphere_z_factor, sphere_radius_factor):
        self.sphere_radius_factor = sphere_radius_factor
        self.sphere_z_factor = sphere_z_factor
        self.sphere_y_factor = sphere_y_factor
        self.sphere_x_factor = sphere_x_factor

    def __call__(self, pil_img):
        """
        :param pil_img: PIL Image
        :return: numpy.ndarray of image, numpy.ndarray with all lifted 3d coord points on the sphere.
         shape: (pil_img.shape[0], pil_img.shape[1], 3)
        """
        img = np.array(pil_img)
        img_w = img.shape[1]
        img_h = img.shape[0]
        sphere_x = img_h * self.sphere_x_factor
        sphere_y = img_w * self.sphere_y_factor
        sphere_z = max(img_h, img_w) * self.sphere_z_factor
        sphere_radius = max(img_h, img_w) * self.sphere_radius_factor
        lifted_points = lift_image(img, sphere_x, sphere_y, sphere_z, sphere_radius)
        return {'image': img, 'coordinates': lifted_points}


class NormalizeTransform(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']       # np.array
        img_tensor = transforms.ToTensor()(img)
        img_tensor_norm = transforms.Normalize(mean=self.mean, std=self.std)(img_tensor)
        img_norm = img_tensor_norm.cpu().data.numpy().transpose(1, 2, 0)
        return {'image': img_norm, 'coordinates': sample['coordinates']}


class ToLatticeImageTransform(object):
    def __init__(self, depth_factor=1, disc_fn=naive_discretize_to_lattice):
        self.disc_fn = disc_fn
        self.depth_factor = depth_factor

    def __call__(self, sample):
        img, lifted_points = sample['image'], sample['coordinates']
        cube = self.disc_fn(img, lifted_points, self.depth_factor)
        return cube


class NormalizeCubeTransform(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, cube):
        # TODO: add support of several channels
        cube_normed = (cube - self.mean) / self.std
        return cube_normed


class ToTensor3DTransform(object):

    def __call__(self, cube):
        tensor_cube = from_numpy(cube).type(FloatTensor).permute(3, 0, 1, 2)
        return tensor_cube

import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)