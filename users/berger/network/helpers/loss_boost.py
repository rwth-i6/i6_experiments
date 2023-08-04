from typing import Dict


def add_loss_boost(
    network: Dict,
    boost_positions_mask: str,
    name: str = "boosted_loss",
    output_layer: str = "output",
    scale: float = 5.0,
) -> str:
    network["ce_loss"] = {
        "class": "loss",
        "from": output_layer,
        "loss_": "ce",
    }

    network[name] = {
        "class": "eval",
        "from": ["ce_loss", boost_positions_mask],
        "eval": f'self.network.get_config().typed_value("loss_boost_func")(source(0), source(1))',
        "loss": "as_is",
        "loss_opts": {"scale": scale},
    }

    return name


# def loss_boost_func(loss, boost_positions_mask):
#     import tensorflow as tf

#     blanks = tf.where(
#         boost_positions_mask,
#         tf.zeros_like(loss, dtype=tf.float32),
#         tf.ones_like(loss, dtype=tf.float32),
#     )
#     blank_count = tf.math.maximum(1.0, tf.reduce_sum(blanks, axis=0, keepdims=True))

#     downscaled_loss = loss / blank_count
#     final_loss = tf.where(boost_positions_mask, loss, downscaled_loss)

#     return final_loss


def loss_boost_func(loss, boost_positions_mask):
    import tensorflow as tf

    return tf.where(boost_positions_mask, loss, tf.zeros_like(loss))
