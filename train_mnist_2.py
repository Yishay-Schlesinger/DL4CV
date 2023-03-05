import torch
import os
import numpy as np
from time import time
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from escnn.nn import GeometricTensor
from escnn.group import groups

from data_loader import DatasetLoader
from data_utils import ToSphereTransform, ToLatticeImageTransform,\
    ToTensor3DTransform, NormalizeTransform, apply_mobius_by_sphere_rot, plot_cube, plot_lifted_points
from steerable_models import SO3SteerableCNN, SO3SteerableTinyCNN
from conventional_models import TinyCNN


CHECKPOINT_PATH = "saved_models/steerable_models/mnist"


def model_test_equi(model: torch.nn.Module, total_steps=5):
    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(tqdm(test_loader, total=total_steps)):
            # x, t = next(iter(test_loader))

            x_og = x.to(device)
            y_og = model(x_og)
            _, prediction = torch.max(y_og.data, 1)
            for j, g in enumerate(groups.so3_group().testing_elements()):
                # TODO: add support for 2d images:
                # lift to sphere -> apply g -> project back
                x_transformed = GeometricTensor(x, SO3Tiny_model.input_type).transform(g).tensor.to(device)
                y_trans = model(x_transformed)
                _, prediction_trans = torch.max(y_trans.data, 1)
                if not torch.allclose(y_og, y_trans, atol=1e-5):
                    print(f"Failed for element {j}; mean error: {(y_og - y_trans).abs().mean()}")


def model_test_mobius():
    with torch.no_grad():
        Tiny_model.eval()
        SO3Tiny_model.eval()
        phi = -np.pi / 2
        theta = -np.pi / 8
        psi = np.pi / 2
        # x = np.random.normal(loc=0.1307, scale=0.3081, size=(28, 28, 1))
        dataset = datasets.MNIST(root="datasets_torchvision/mnist", train=True, download=True, transform=None)
        x, t = dataset[1]
        x = np.array(x).reshape((28, 28, 1))
        _, x_mobius = apply_mobius_by_sphere_rot(x, x.shape[0]//2, x.shape[1]//2, x.shape[0]//2, x.shape[0]//2, phi, theta, psi)

        conv_transform_fn = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))])
        so3_transform_fn = transforms.Compose([ToSphereTransform(0.5, 0.5, 0.5, 0.5),
                                               NormalizeTransform((0.1307,), (0.3081,)),
                                               ToLatticeImageTransform(),
                                               ToTensor3DTransform()])
        x_conv = conv_transform_fn(x).to(device)
        x_conv = x_conv.reshape(1, 1, 28, 28).type(torch.FloatTensor).to(device)
        x_mobius_conv = conv_transform_fn(x_mobius).to(device)
        x_mobius_conv = x_mobius_conv.reshape(1, 1, 28, 28).type(torch.FloatTensor).to(device)

        x_so3 = so3_transform_fn(x).to(device)
        x_so3 = x_so3.reshape(1, 1, 28, 28, 28).type(torch.FloatTensor).to(device)
        x_mobius_so3 = so3_transform_fn(x_mobius).to(device)
        x_mobius_so3 = x_mobius_so3.reshape(1, 1, 28, 28, 28).type(torch.FloatTensor).to(device)

        g = groups.so3_group().element([phi, theta, psi], param='ZXZ')
        x_trans_so3 = GeometricTensor(x_so3, SO3Tiny_model.input_type).transform(g).tensor.to(device)

        def indices_condition(indices):
            d = np.linalg.norm(indices - np.array([x.shape[0] // 2, x.shape[0] // 2, x.shape[0] // 2]), axis=-1)
            c1 = (x.shape[0] // 2 - 2 < d)
            c2 = (d < x.shape[0] // 2 + 2)
            return c1 * c2
        plot_cube(np.transpose(x_trans_so3[0].data.cpu().numpy(), (1, 2, 3, 0)), lambda c: c > 0.05, indices_condition, 'gray')

        y_conv = Tiny_model(x_conv)
        y_mob_conv = Tiny_model(x_mobius_conv)

        y_so3 = SO3Tiny_model(x_so3)
        y_so3_mob = SO3Tiny_model(x_mobius_so3)
        print("=====================================================")
        print(f"Conventional: {(y_conv - y_mob_conv).abs().mean()}")
        print(f"Steerable   : {(y_so3 - y_so3_mob).abs().mean()}")


def model_test(model: torch.nn.Module, ):
    total_steps = len(test_loader)
    total = 0
    correct = 0

    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(tqdm(test_loader, total=total_steps)):
            x = x.to(device)
            t = t.to(device)

            y = model(x)

            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()
            del x, y, t
            # if i == 250:
            #     break
    return correct/total*100.


def train(model: torch.nn.Module, lr=1e-4, wd=1e-4, checkpoint_path: str = None, num_epochs: int = 21):
    if checkpoint_path is not None:
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_path)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in tqdm(range(num_epochs)):
        total_steps = len(train_loader)
        model.train()
        for i, (x, t) in enumerate(tqdm(train_loader, total=total_steps)):
            optimizer.zero_grad()

            x = x.to(device)
            t = t.to(device)
            # start_forward = time()
            y = model(x)
            # print("done forward. took: ", time() - start_forward)
            loss = loss_function(y, t)

            loss.backward()

            optimizer.step()
            loss_value = loss.data.cpu().numpy()
            del x, y, t, loss
            if i % 10 == 0:
                print("step ", i, "/", total_steps, " | loss: ", loss_value)

        # if epoch % 1 == 0:
        #     accuracy = model_test(model)
        #     print(f"epoch {epoch} | test accuracy: {accuracy}")
    accuracy = model_test(model)
    print(f"test accuracy: {accuracy}")

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    data_root = "datasets_torchvision/mnist"
    dataset_name = 'MNIST'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    start_prepare_data = time()
    print("==== preparing data ====")
    train_transform_fn = transforms.Compose([ToSphereTransform(0.5, 0.5, 0.5, 0.5),
                                             NormalizeTransform((0.1307,), (0.3081,)),
                                             ToLatticeImageTransform(),
                                             ToTensor3DTransform()])

    train_transform_fn_conventional = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))])
    dataset_loader = DatasetLoader(dataset_name=dataset_name, data_root=data_root,
                                   train_batch_size=[16, 16, 16, 16], test_batch_size=[16, 16, 16, 16], device=device,
                                   train_transform_fn=train_transform_fn_conventional,
                                   test_transform_fn=train_transform_fn_conventional,
                                   train_targets=[0, 1, 3, 4],
                                   test_targets=[0, 1, 3, 4])
    train_loader, test_loader = dataset_loader.get_data_loaders()
    print("Done. took: ", time() - start_prepare_data)
    curr_time = time()

    print("==== creating model ====")
    # SO3_model = SO3SteerableCNN(10)
    # SO3_model = SO3_model.to(device)

    SO3Tiny_model = SO3SteerableTinyCNN(4)
    SO3Tiny_model = SO3Tiny_model.to(device)

    Tiny_model = TinyCNN(4)
    Tiny_model = Tiny_model.to(device)

    print("Done. took: ", time() - curr_time)
    print("==== staring training ====")
    train(Tiny_model, checkpoint_path="conventional_tiny-mnist_t0134.ckpt", num_epochs=3)
    train(SO3Tiny_model, checkpoint_path="steerable_so3tiny-mnist_t0134.ckpt", num_epochs=3)
    print("==== staring testing ====")
    model_test_mobius()
    # accuracy = model_test(Tiny_model)
    # print(f"Test accuracy: {accuracy}")
    #
    # model_test_equi(Tiny_model)
    # torch.manual_seed(42)


