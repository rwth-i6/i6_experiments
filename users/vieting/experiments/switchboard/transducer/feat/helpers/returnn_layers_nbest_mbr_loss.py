import tensorflow as tf
from returnn.tf.layers.basic import _ConcatInputLayer, register_layer_class
from returnn.tf.util.data import Data


# iterative computation over NBest to solve memory issue (still N limit due to gradients)
# also similar batch settings as in fullsum training can be used
class IterativeNBestMBRLossLayer(_ConcatInputLayer):
    layer_class = "iterative_nbest_mbr"

    def __init__(
        self,
        sources,
        nbest,
        use_nbest_score=False,
        renorm_scale=1.0,
        rnnt_loss_scale=0,
        blank_label=0,
        input_type="logit",
        reuse_joint=("_joint", 1024, "tanh"),
        reuse_output=("_output", 79),
        **kwargs,
    ):
        super().__init__(**kwargs)
        enc_out = sources[0].output  # encoder output of B utterances
        dec_out = sources[1].output  # predNN output of B*N NBest seqs
        seq_out = sources[2].output  # NBest seqs
        enc = enc_out.get_placeholder_as_batch_major()
        dec = dec_out.get_placeholder_as_batch_major()
        seq = seq_out.get_placeholder_as_batch_major()
        enc_lens = enc_out.get_sequence_lengths()
        dec_lens = dec_out.get_sequence_lengths()
        seq_lens = seq_out.get_sequence_lengths()
        # additional scores used in NBest generation, e.g. lm_score
        add_scores = sources[3].output.get_placeholder_as_batch_major()  # (B*N,)
        nbest_risks = sources[4].output.get_placeholder_as_batch_major()
        # all further input regarded as computation dependency for memory issue
        if len(sources) >= 6:
            extra_dep = [s.output.placeholder for s in sources[5:]]
        else:
            extra_dep = []

        import tensorflow as tf
        import numpy as np
        from returnn.tf.util import dot, reuse_name_scope, dropout, get_activation_function
        from returnn.extern_private.BergerMonotonicRNNT import rnnt_loss

        B = tf.shape(enc)[0]
        BN = tf.shape(add_scores)[0]
        F1 = enc.shape[-1]
        F2 = dec.shape[-1]

        # reuse joint and output layers params for computation
        # same param_name under different reuse_layer scope: use param_name+suffix for map
        # fixed format (e.g. default args): param_name_suffix, dim, activation
        assert len(reuse_joint) >= 3 and len(reuse_output) >= 2
        with self.var_creation_scope(param_name_suffix=reuse_joint[0]):
            W_joint = self.add_param(
                tf.get_variable(name="W", shape=(F1 + F2, reuse_joint[1]), dtype=tf.float32),
                param_name_suffix=reuse_joint[0],
            )
            b_joint = self.add_param(
                tf.get_variable(name="b", shape=(reuse_joint[1],), dtype=tf.float32),
                param_name_suffix=reuse_joint[0],
            )
        joint_act_func = get_activation_function(reuse_joint[2])
        with self.var_creation_scope(param_name_suffix=reuse_output[0]):
            W_output = self.add_param(
                tf.get_variable(name="W", shape=(reuse_joint[1], reuse_output[1]), dtype=tf.float32),
                param_name_suffix=reuse_output[0],
            )
            b_output = self.add_param(
                tf.get_variable(name="b", shape=(reuse_output[1],), dtype=tf.float32),
                param_name_suffix=reuse_output[0],
            )
        # no L2 on output layer
        self.no_l2_params = ["W" + reuse_output[0], "b" + reuse_output[0]]

        # inner loop over each seq for no-zero-padding concatenation
        # possible dynamic-NBest: empty seqs will have negligible computation cost
        def iterative_compress_concat(dec_B, dec_lens_B):
            b = tf.constant(0, dtype=tf.int32)
            cond = lambda b, _: tf.less(b, B)
            out = tf.TensorArray(
                dtype=tf.float32,
                size=B,
                dynamic_size=False,
                element_shape=[None, F1 + F2],
                infer_shape=False,
                clear_after_read=True,
            )

            def body(b, out):
                encT = tf.pad(
                    enc[b, : enc_lens[b]],
                    [(0, 0), (0, F2)],
                    mode="CONSTANT",
                    constant_values=0,
                )  # (T, F)
                decU = tf.pad(
                    dec_B[b, : dec_lens_B[b]],
                    [(0, 0), (F1, 0)],
                    mode="CONSTANT",
                    constant_values=0,
                )  # (U+1, F)
                return b + 1, out.write(
                    b,
                    tf.reshape(
                        tf.expand_dims(encT, 1) + tf.expand_dims(decU, 0),
                        [enc_lens[b] * dec_lens_B[b], F1 + F2],
                    ),
                )  # (T * U+1, F)

            b, out = tf.while_loop(cond, body, [b, out])
            return out.concat()  # (sum_B T_b * U_b+1, F)

        # iterative computation: B seqs/iteration with gap N (deactivate parallelization by dependency)
        score_list = []
        for n in range(nbest):
            with tf.control_dependencies(score_list + extra_dep):
                indices = tf.range(n, BN, delta=nbest)
                dec_B = tf.gather(dec, indices, axis=0)
                dec_lens_B = tf.gather(dec_lens, indices, axis=0)
                compress_concat = iterative_compress_concat(dec_B, dec_lens_B)
                # forward jointNN
                compress_concat = self.network.cond_on_train(
                    lambda: dropout(
                        compress_concat,
                        keep_prob=1 - self.dropout,
                        noise_shape=(1, F1 + F2),
                        seed=self.network.random.randint(2**31),
                    ),
                    lambda: compress_concat,
                )
                joint = tf.add(dot(compress_concat, W_joint), b_joint, name="add_bias")
                if joint_act_func:
                    joint = joint_act_func(joint)
                output = tf.add(dot(joint, W_output), b_output, name="add_bias")
                # compute rnnt fullsum loss
                seq_B = tf.gather(seq, indices, axis=0)
                seq_lens_B = tf.gather(seq_lens, indices, axis=0)
                scores = rnnt_loss(
                    output,
                    seq_B,
                    enc_lens,
                    seq_lens_B,
                    blank_label=blank_label,
                    input_type=input_type,
                )
                scores.set_shape((None,))  # (B,)
                score_list.append(tf.scatter_nd(tf.expand_dims(indices, -1), scores, [BN]))

        # -log(P_rnnt) for all NBest seqs (B*N,)
        nbest_scores = tf.add_n(score_list)  # ensure dependency
        # ground-truth seqs mask
        ref_mask = tf.where(
            tf.equal(nbest_risks, 0),
            tf.ones_like(nbest_risks),
            tf.zeros_like(nbest_risks),
        )
        ref_mask = tf.where(tf.greater(seq_lens, 0), ref_mask, tf.zeros_like(ref_mask))
        # optional scaled rnnt-loss
        if rnnt_loss_scale > 0:
            rnnt_loss = rnnt_loss_scale * tf.reduce_sum(nbest_scores * ref_mask)
        else:
            rnnt_loss = 0
        # add additional score and gobal scale before renormalization
        if use_nbest_score:
            nbest_scores = nbest_scores + add_scores
        nbest_scores = nbest_scores * renorm_scale
        # possible dynamic-NBest: filter out empty seqs
        empty_scores = tf.ones_like(nbest_scores) * np.inf
        nbest_scores = tf.where(tf.equal(seq_lens, 0), empty_scores, nbest_scores)
        # renormalization within each (dynamic) NBest
        norm_nbest = tf.reshape(nbest_scores, [-1, nbest])  # (B, N)
        norm_nbest = tf.nn.softmax(-norm_nbest, axis=-1)
        # multiply risk of each hyp (explicit zero-out empty seqs to be sure)
        nbest_mbr = tf.reshape(norm_nbest, [-1]) * nbest_risks
        nbest_mbr = tf.where(tf.equal(seq_lens, 0), tf.zeros_like(nbest_mbr), nbest_mbr)
        # directly a scalar MBR loss + optional scaled rnnt-loss
        self.output.placeholder = tf.reduce_sum(nbest_mbr) + rnnt_loss

    @classmethod
    def get_out_data_from_opts(cls, name, sources, **kwargs):
        from returnn.tf.util.data import Data

        return Data(
            name="%s_output" % name,
            shape=(),
            dtype="float32",
            batch_dim_axis=None,
            time_dim_axis=None,
            feature_dim_axis=None,
        )

    def get_params_l2_norm(self):
        import tensorflow as tf

        return 2 * sum(
            [tf.nn.l2_loss(param) for (name, param) in sorted(self.params.items()) if not name in self.no_l2_params]
        )


register_layer_class(IterativeNBestMBRLossLayer)
