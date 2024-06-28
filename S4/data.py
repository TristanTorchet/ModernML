import jax
import numpy as np
import torch
from jax import numpy as jnp
from torch.utils.data import TensorDataset
from tqdm import tqdm


def create_sin_x_dataset(n_examples=1024, bsz=128):
    print("[*] Generating Toy Dataset: sin(x)...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 16, 8, 1
    x = np.linspace(0, 2 * np.pi, num=SEQ_LENGTH)
    # jnp.digitize: Return the indices of the bins to which each value in input array belongs.
    y = np.digitize(np.sin(x), np.linspace(-1, 1, num=N_CLASSES))

    # Tile this `n_examples` times...
    # data will be a tensor of shape (n_examples, SEQ_LENGTH, IN_DIM)
    data = torch.Tensor(
        np.tile(
            np.expand_dims(np.expand_dims(y, -1), 0), reps=[n_examples, 1, 1]
        )
    )

    # Build Datasets -- Two entries to match (inputs, targets) structure
    train = TensorDataset(data, data)
    test = TensorDataset(data[:1], data[:1])

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=1, shuffle=False # TODO: I changed the batch size from bsz to 1 as the test is only one example
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, data


def create_sin_ax_b_dataset(n_examples=2000, bsz=256): # TODO: n_examples=20000 (in original code)
    print("[*] Generating sin(ax + b) Dataset...")

    # Constants – `a` sampled uniform from [1, 10], `b` sampled uniform [0, 5]
    SEQ_LENGTH, N_CLASSES, IN_DIM, A_MAX, B_MAX = 16, 8, 1, 10, 5 # TODO SEQ_LENGTH=16000 in original code
    train_data, test_data = [], []
    data_key = jax.random.PRNGKey(21)

    # Loop through `n_examples` and generate data
    print(f"\t=>> Generating {n_examples} Training Examples...")
    x = np.linspace(0, 2 * np.pi, num=SEQ_LENGTH)
    for _ in tqdm(range(n_examples)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(
            a_rng, minval=1.0, maxval=A_MAX
        ), jax.random.uniform(b_rng, maxval=B_MAX)
        train_data.append(
            np.digitize(np.sin(a * x + b), np.linspace(-1, 1, num=N_CLASSES))
        )

    # Generate 1 Batch of Test Examples
    print(f"\t=>> Generating {bsz} Test Examples...")
    for _ in tqdm(range(bsz)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(
            a_rng, minval=1.0, maxval=A_MAX
        ), jax.random.uniform(b_rng, maxval=B_MAX)
        test_data.append(
            np.digitize(np.sin(a * x + b), np.linspace(-1, 1, num=N_CLASSES))
        )

    # Build Datasets - Two entries to match (inputs, targets) structure
    train_data = torch.Tensor(np.expand_dims(np.array(train_data), -1))
    test_data = torch.Tensor(np.expand_dims(np.array(test_data), -1))
    train = TensorDataset(train_data, train_data)
    test = TensorDataset(test_data, test_data)

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False, drop_last=True
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, (jnp.array(train_data), jnp.array(test_data))
