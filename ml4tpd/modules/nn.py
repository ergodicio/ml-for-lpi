from typing import Dict, Callable
from jax import numpy as jnp, random as jr, Array
import equinox as eqx


class PRNGKeyArray:
    def __init__(self, key: Array):
        self.key = key


def get_activation(activation: str) -> Callable:
    """
    Get the activation function from the string name.

    Args:
        activation (str): The name of the activation function.

    Returns:
        Callable: The activation function
    """
    if activation == "tanh":
        activation = jnp.tanh
    elif activation == "relu":
        from jax.nn import relu as activation
    elif activation == "sigmoid":
        from jax.nn import sigmoid as activation
    elif activation == "softplus":
        from jax.nn import softplus as activation
    elif activation == "elu":
        from jax.nn import elu as activation
    elif activation == "leaky_relu":
        from jax.nn import leaky_relu as activation

    else:
        raise NotImplementedError(f"Activation function {activation} not recognized.")

    return activation


class GenerativeModel(eqx.Module):
    amp_decoder: eqx.Module
    phase_decoder: eqx.Module
    output_width: int

    def __init__(
        self,
        decoder_width: int,
        decoder_depth: int,
        input_width: int,
        output_width: int,
        key: int,
        activation: str = "tanh",
    ):
        super().__init__()
        da_k, dp_k = jr.split(jr.PRNGKey(key), 2)

        act_fun = get_activation(activation)

        self.amp_decoder = eqx.nn.MLP(
            input_width, output_width, width_size=decoder_width, depth=decoder_depth, key=da_k, activation=act_fun
        )
        self.output_width = output_width
        self.phase_decoder = eqx.nn.MLP(
            input_width, output_width, width_size=decoder_width, depth=decoder_depth, key=dp_k, activation=act_fun
        )

    def __call__(self, x: Array) -> Dict:
        amps = 3 * self.amp_decoder(x)
        phases = 3 * self.phase_decoder(x)
        return {"amps": amps, "phases": phases}
