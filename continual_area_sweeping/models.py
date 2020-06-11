import tensorflow as tf
from tensorflow import layers
import tensorflow.contrib.layers as contrib_layers


def original_dqn_conv(inpt):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope("explorerl", reuse=False):
        out = inpt
        #print (out.shape)
        out = layers.conv2d(out,
                            filters=16,
                            #kernel_size=8,
                            kernel_size=5,
                            #strides=(4, 4),
                            strides=1,
                            #padding='SAME',
                            activation=tf.nn.relu)
        #print (out.shape)
        out = layers.conv2d(out,
                            filters=32,
                            #kernel_size=4,
                            kernel_size=3,
                            #strides=(2, 2),
                            strides=1,
                            #padding='SAME',
                            activation=tf.nn.relu)
        #print (out.shape)
        return out

def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            latent = network(input_placeholder)
            if isinstance(latent, tuple):
                if latent[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent = latent[0]

            latent = contrib_layers.flatten(latent)

            with tf.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    action_out = contrib_layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        action_out = contrib_layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = contrib_layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = contrib_layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            state_out = contrib_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    state_score = contrib_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
            return q_out

    return q_func_builder