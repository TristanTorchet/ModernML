import jax
import jax.numpy as jnp
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve
from functools import partial

def discretize(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, step: float):
    '''
    Discretize continuous-time state-space model. We use the trapezoidal rule (aka Bilinear, different from ZOH).
    :param A: jnp.ndarray (N, N), state transition matrix
    :param B: jnp.ndarray (N, 1), input matrix
    :param C: jnp.ndarray (1, N), output matrix
    :param step: float, time step of the discretization
    :return:
        Ab: jnp.ndarray (N, N), discretized state transition matrix
        Bb: jnp.ndarray (N, 1), discretized input matrix
        C: jnp.ndarray (1, N), output matrix
    '''
    I = jnp.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        print(f'u_k:{u_k.shape}')
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)



def K_conv(Ab: jnp.array, Bb: jnp.array, Cb: jnp.array, L: int):
    '''
    Computes the convolution kernel for the SSM.
    K = (CB, CAB, C(A^2)B, ..., C(A^{L-1})B)
    WARNING: this is a naive implementation and should be avoided for large L (stability issues).
    :param Ab: Discretized state transition matrix A
    :param Bb: Discretized input matrix B
    :param Cb: Discretized output matrix C
    :param L: Sequence length
    :return:
    '''
    return jnp.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )

def causal_convolution(u: jnp.array, K: jnp.array, nofft: bool = False):
    '''
    Computes the convolution of the input signal u with the kernel K.
    Can FFT for faster computation in the frequency domain (need to pad the u and K to the same length).
    :param u: full length input signal
    :param K: SSM Kernel
    :param nofft: binary flag to disable FFT
    :return: full length output signal
    '''
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]] # from jax.scipy.signal
    else:
        assert K.shape[0] == u.shape[0]
        ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
        Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return jnp.fft.irfft(out)[: u.shape[0]]


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -jnp.sum(one_hot_label * logits)


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.argmax(logits) == label