import functools

import jax
from flax import linen as nn
from jax import numpy as jnp


class ScannedLSTM(nn.Module):
    hidden_size: int = 64

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.LSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        c, h = carry
        ins, resets = x

        reset_c, reset_h = self.initialize_carry(ins.shape[0], self.hidden_size)
        c = jnp.where(resets[:, jnp.newaxis], reset_c, c)
        h = jnp.where(resets[:, jnp.newaxis], reset_h, h)
        lstm_state = (c, h)

        new_lstm_state, y = nn.LSTMCell(features=self.hidden_size)(lstm_state, ins)
        return new_lstm_state, y


class LSTMEncoder(nn.Module):
    hidden_size: int = 64
    output_size: int = 20

    @nn.compact
    def __call__(self, hidden, x):
        hidden, x = ScannedLSTM(self.hidden_size)(hidden, x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        out = nn.Dense(features=self.output_size)(x)
        out = nn.sigmoid(out)
        return hidden, out


class ObsActionDecoder(nn.Module):
    hidden_size: int = 64
    output_size_1: int = 32
    output_size_2: int = 6

    @nn.compact
    def __call__(self, x):
        x1 = nn.Dense(features=self.hidden_size)(x)
        x1 = nn.relu(x1)
        x1 = nn.Dense(features=self.hidden_size)(x1)
        x1 = nn.relu(x1)
        out1 = nn.Dense(features=self.output_size_1)(x1)

        x2 = nn.Dense(features=self.hidden_size)(x)
        x2 = nn.relu(x2)
        x2 = nn.Dense(features=self.hidden_size)(x2)
        x2 = nn.relu(x2)
        out2 = nn.Dense(features=self.output_size_2)(x2)
        out2 = nn.softmax(out2, axis=-1)

        return out1, out2


class ActionDecoder(nn.Module):
    hidden_size: int = 64
    output_size: int = 6

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        out = nn.Dense(features=self.output_size)(x)
        out = nn.softmax(out, axis=-1)
        return out
