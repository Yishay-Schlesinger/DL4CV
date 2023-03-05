import torch
import numpy as np
from escnn.group import groups
from escnn import nn, gspaces
from torchvision.transforms import ToTensor
from torch import from_numpy, FloatTensor

from data_utils import get_mean_std_of_cube_data, ToSphereTransform,\
    naive_discretize_to_lattice, rotate_image_points_on_sphere, naive_discretize_with_blurr, spread_discretize

np.random.seed(1)


def cube_equivariance_on_random_image(disc_fn, lifted_dict):
    """ test if disc(g.lifted_points)=g.disc(lifted_points) for g in SO(3)"""
    img, lifted_points = lifted_dict['image'], lifted_dict['coordinates']
    depth_factor = 1
    cube = disc_fn(img, lifted_points, depth_factor)
    tetra_errors = []
    errors = []
    for j, g in enumerate(groups.so3_group().testing_elements()):
        euler_angles = g.to('ZXZ')
        phi, theta, psi = euler_angles[0], euler_angles[1], euler_angles[2]
        # apply g on cube:
        tensor_cube = from_numpy(cube).type(FloatTensor)
        geo_cube = feat_type_in(tensor_cube.reshape(1, 1, W, W, W))
        geo_g_disc = geo_cube.transform(g)
        g_disc = geo_g_disc.tensor.reshape(W, W, W, 1)
        # apply g on points then discretize:
        g_points = rotate_image_points_on_sphere(lifted_points, W//2, W//2, W//2, phi, theta, psi)
        np_disc_g = disc_fn(img, g_points, depth_factor)
        disc_g = from_numpy(np_disc_g).type(FloatTensor).reshape(W, W, W, 1)
        # compute error:
        relative_error = torch.norm(disc_g - g_disc).item() / torch.norm(tensor_cube).item()
        if j < 12:
            tetra_errors.append(relative_error)
        errors.append(relative_error)
        print(f"Error for element {j}: ", relative_error)
    return np.array(tetra_errors).mean(), np.array(errors).mean()


# def mobius_equivariance_on_random_image(disc_fn, lifted_dict):
#     return mob_tetra_error, mob_ico_error


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    disc_functions = [naive_discretize_to_lattice]
    r3_act = gspaces.rot3dOnR3(maximum_frequency=3)
    G = r3_act.fibergroup
    feat_type_in = nn.FieldType(r3_act, [G.trivial_representation])
    W_img = 29
    P = 14
    W = W_img + 2 * P
    img_mean, img_std = 0.1307, 0.3081
    image = np.random.normal(img_mean, img_std, size=(W_img, W_img, 1))
    padded_image = np.zeros((W, W, 1))
    padded_image[P:P+W_img, P:P+W_img] = image
    # image = torch.normal(img_mean, img_std, size=(W, W, 1))
    cube_mean, cube_std = get_mean_std_of_cube_data(img_mean, img_std, W_img * W_img, W * W * W - W_img*W_img, 0.)
    lift_fn = ToSphereTransform(0.5, 0.5, 0.5, 0.5)
    res_dict = lift_fn(padded_image)
    for i, f in enumerate(disc_functions):
        tetra_res, ico_res = cube_equivariance_on_random_image(f, res_dict)
        print("============= test mobius equivariance =========")
        # mob_tetra_res, mob_ico_res = mobius_equivariance_on_random_image(f, res_dict)
        print(f"============ model {i} =============")
        print("mean tetra error: ", tetra_res)
        print("mean ico error: ", ico_res)
        # print("mean mob tetra error: ", mob_tetra_res)
        # print("mean mob ico error: ", mob_ico_res)
