import numpy as np
from torchvision import transforms
from PIL import Image

from data_utils import apply_mobius_by_sphere_rot, lift_image, rotate_image_points_on_sphere


""" Test difference between L(mob_g(I)) and g^-1.L(I)"""


img_path = 'example_img1.png'
og_img = Image.open(img_path).convert('RGB')
img = transforms.Resize((128, 128))(og_img)
img = np.array(img)
phi = -np.pi/2
theta = -np.pi/8
psi = np.pi/2
#   Mob(I):
trans_img, trans_img_int = apply_mobius_by_sphere_rot(img, img.shape[0]//2, img.shape[1]//2, img.shape[0]//2, img.shape[0]//2, phi, theta, psi)
#   L(Mob(I)):
lift_trans_img = lift_image(trans_img_int, img.shape[0]//2, img.shape[1]//2, img.shape[0]//2, img.shape[0]//2)
#   g^-1.L(Mob(I)):
lift_trans_img_rot = rotate_image_points_on_sphere(lift_trans_img, img.shape[0]//2, img.shape[1]//2, img.shape[0]//2, -psi, -theta, -phi)
#   L(I)
lift_img = lift_image(img, img.shape[0]//2, img.shape[1]//2, img.shape[0]//2, img.shape[0]//2)
#   g^-1.L(I):
lift_img_rot = rotate_image_points_on_sphere(lift_img, img.shape[0]//2, img.shape[1]//2, img.shape[0]//2, -psi, -theta, -phi)
print("done")
