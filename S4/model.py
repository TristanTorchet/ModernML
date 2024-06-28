import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal
from utils import discretize, K_conv, scan_SSM, causal_convolution


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
                jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return init



class SSMLayer(nn.Module):
    N: int  # State dimension
    l_max: int  # Sequence length
    decode: bool = False  # if True, use RNN mode

    def setup(self):
        # SSM parameters
        self.A = self.param("A", lecun_normal(), (self.N, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = jnp.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", jnp.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u



def cloneLayer(layer):
    '''
    Clone a layer in parallel with the same parameters and cache.

    :param layer:
    :return:
    '''
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


SSMLayer = cloneLayer(SSMLayer)  # Redefine SSMLayer as stacked layers




class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float  # Dropout rate
    d_model: int  # Dimension of the model, i.e. N
    prenorm: bool = True  # If True, use pre-normalization (which is a technique to improve convergence)
    glu: bool = True  # If True, use Gated Linear Units as activation of the inner layer
    training: bool = True  # If True, use dropout
    decode: bool = False  # If True, use RNN mode

    def setup(self):
        ## **layer unpacks the dictionary into keyword arguments
        self.seq = self.layer_cls(**self.layer, decode=self.decode)  # SSMLayer
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x



class Embedding(nn.Embed):
    '''

    '''
    num_embeddings: int  # Number of embeddings from a single input
    features: int  # Number of features of the embedding

    @nn.compact
    def __call__(self, x):
        # x[..., 0] is the first element of the input tensor x (instead of writing x[:, :, 0] for example in 3D)
        # y is the embedding of the first element of the input tensor x
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return jnp.where(x > 0, y, 0.0)



class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Extra arguments to pass into layer constructor
    d_output: int  # Output dimension
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False  # Use nn.Embed instead of nn.Dense encoder
    classification: bool = False
    training: bool = True
    decode: bool = False  # Probably should be moved into layer_args

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0  # Normalize
            if not self.decode:
                x = jnp.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = jnp.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)



# In Flax we add the batch dimension as a lifted transformation.
# We need to route through several variable collections which handle RNN and parameter caching (described below).

BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)