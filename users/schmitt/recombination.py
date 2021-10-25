

def targetb_recomb_train(layer, batch_dim, scores_in, scores_base, base_beam_in, end_flags, **kwargs):
  """
  :param ChoiceLayer layer:
  :param tf.Tensor batch_dim: scalar
  :param tf.Tensor scores_base: (batch,base_beam_in,1). existing beam scores
  :param tf.Tensor scores_in: (batch,base_beam_in,dim). log prob frame distribution
  :param tf.Tensor end_flags: (batch,base_beam_in)
  :param tf.Tensor base_beam_in: int32 scalar, 1 or prev beam size
  :rtype: tf.Tensor
  :return: (batch,base_beam_in,dim), combined scores
  """
  import tensorflow as tf
  from returnn.tf.util.basic import where_bc, nd_indices, tile_transposed
  scores = scores_in + scores_base  # (batch,beam,dim)
  dim = layer.output.dim

  u = layer.explicit_search_sources[0].output  # prev:u actually. [B*beam], pos in target [0..decT-1]
  assert u.shape == ()
  u_t = tf.reshape(tf.reshape(u.placeholder, (batch_dim, -1))[:, :base_beam_in],
                   (-1,))  # u beam might differ from base_beam_in
  targets = layer.network.parent_net.extern_data.data[eval("target")]  # BPE targets, [B,decT]
  assert targets.shape == (None,) and targets.is_batch_major
  target_lens = targets.get_sequence_lengths()  # [B]
  target_lens_exp = tile_transposed(target_lens, axis=0, multiples=base_beam_in)  # [B*beam]
  missing_targets = target_lens_exp - u_t  # [B*beam]
  allow_target = tf.greater(missing_targets, 0)  # [B*beam]
  targets_exp = tile_transposed(targets.placeholder, axis=0, multiples=base_beam_in)  # [B*beam,decT]
  targets_u = tf.gather_nd(targets_exp, indices=nd_indices(where_bc(allow_target, u_t, 0)))  # [B*beam]
  targets_u = tf.reshape(targets_u, (batch_dim, base_beam_in))  # (batch,beam)
  allow_target = tf.reshape(allow_target, (batch_dim, base_beam_in))  # (batch,beam)

  t_t = layer.network.get_rec_step_index() - 1  # scalar
  inputs = layer.network.parent_net.get_layer("encoder").output  # encoder, [B,encT]
  input_lens = inputs.get_sequence_lengths()  # [B]
  input_lens_exp = tile_transposed(input_lens, axis=0, multiples=base_beam_in)  # [B*beam]
  allow_blank = tf.less(missing_targets, input_lens_exp - t_t)  # [B*beam]
  allow_blank = tf.reshape(allow_blank, (batch_dim, base_beam_in))  # (batch,beam)

  dim_idxs = tf.range(dim)[None, None, :]  # (1,1,dim)
  masked_scores = where_bc(
    tf.logical_or(tf.logical_and(tf.equal(dim_idxs, eval("targetb_blank_idx")), allow_blank[:, :, None]),
                  tf.logical_and(tf.equal(dim_idxs, targets_u[:, :, None]), allow_target[:, :, None])), scores,
    float("-inf"))

  return where_bc(end_flags[:, :, None], scores, masked_scores)


def get_filtered_score_op(verbose=False):
  cpp_code = """
  #include "tensorflow/core/framework/op.h"
  #include "tensorflow/core/framework/op_kernel.h"
  #include "tensorflow/core/framework/shape_inference.h"
  #include "tensorflow/core/framework/resource_mgr.h"
  #include "tensorflow/core/framework/resource_op_kernel.h"
  #include "tensorflow/core/framework/tensor.h"
  #include "tensorflow/core/platform/macros.h"
  #include "tensorflow/core/platform/mutex.h"
  #include "tensorflow/core/platform/types.h"
  #include "tensorflow/core/public/version.h"
  #include <cmath>
  #include <map>
  #include <set>
  #include <string>
  #include <tuple>

  using namespace tensorflow;

  REGISTER_OP("GetFilteredScore")
  .Input("prev_str: string")
  .Input("scores: float32")
  //.Input("labels: string")
  .Output("new_scores: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
  });

  class GetFilteredScoreOp : public OpKernel {
  public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
      const Tensor* prev_str = &context->input(0);
      const Tensor* scores = &context->input(1);
      //const Tensor* labels = &context->input(2);

      int n_batch = prev_str->shape().dim_size(0);
      int n_beam = prev_str->shape().dim_size(1);

      Tensor* ret;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n_batch, n_beam}), &ret));
      for(int bat = 0; bat < n_batch; ++bat)
          for(int hyp = 0; hyp < n_beam; ++hyp)
              ret->tensor<float, 2>()(bat, hyp) = scores->tensor<float, 2>()(bat, hyp);

      for(int bat = 0; bat < n_batch; ++bat) {
          std::map<tstring, std::set<int> > new_hyps;  // seq -> set of hyp idx

          for(int hyp = 0; hyp < n_beam; ++hyp) {
              auto& seq_set = new_hyps[prev_str->tensor<tstring, 2>()(bat, hyp)];
              seq_set.insert(hyp);
          }

          for(const auto& items : new_hyps) {
              if(std::get<1>(items).size() > 1) {
                  float best_score = 0.;
                  int best_idx = -1;
                  for(int idx : std::get<1>(items)) {
                      float score = scores->tensor<float, 2>()(bat, idx);
                      if(score > best_score || best_idx == -1) {
                          best_score = score;
                          best_idx = idx;
                      }
                  }

                  float sum_score = 0.;
                  for(int idx : std::get<1>(items)) {
                      float score = scores->tensor<float, 2>()(bat, idx);
                      sum_score += expf(score - best_score);
                  }
                  sum_score = logf(sum_score) + best_score;

                  for(int idx : std::get<1>(items)) {
                      if(idx != best_idx)
                          ret->tensor<float, 2>()(bat, idx) = -std::numeric_limits<float>::infinity();
                      else
                          ret->tensor<float, 2>()(bat, idx) = sum_score;
                  }
              }
          }
      }
  }
  };
  REGISTER_KERNEL_BUILDER(Name("GetFilteredScore").Device(DEVICE_CPU), GetFilteredScoreOp);
  """
  from returnn.tf.util.basic import OpCodeCompiler
  compiler = OpCodeCompiler(base_name="GetFilteredScore", code_version=1, code=cpp_code, is_cpp=True,
                            use_cuda_if_available=False, verbose=verbose)
  tf_mod = compiler.load_tf_module()
  return tf_mod.get_filtered_score


def get_filtered_score_cpp(prev_str, scores):
  """
  :param tf.Tensor prev_str: (batch,beam)
  :param tf.Tensor scores: (batch,beam)
  :param list[bytes] labels: len (dim)
  :return: scores with logsumexp at best, others -inf, (batch,beam)
  :rtype: tf.Tensor
  """
  import tensorflow as tf
  with tf.device("cpu:0"):
    return get_filtered_score_op()(prev_str, scores)


def targetb_recomb_recog(layer, batch_dim, scores_in, scores_base, base_beam_in, end_flags, **kwargs):
  """
  :param ChoiceLayer layer:
  :param tf.Tensor batch_dim: scalar
  :param tf.Tensor scores_base: (batch,base_beam_in,1). existing beam scores
  :param tf.Tensor scores_in: (batch,base_beam_in,dim). log prob frame distribution
  :param tf.Tensor end_flags: (batch,base_beam_in)
  :param tf.Tensor base_beam_in: int32 scalar, 1 or prev beam size
  :rtype: tf.Tensor
  :return: (batch,base_beam_in,dim), combined scores
  """
  import tensorflow as tf

  prev_str = layer.explicit_search_sources[0].output  # [B*beam], str
  prev_str_t = tf.reshape(prev_str.placeholder, (batch_dim, -1))[:, :base_beam_in]

  # Pre-filter approx (should be much faster), sum approx (better).
  scores_base = tf.reshape(
    get_filtered_score_cpp(prev_str_t, tf.reshape(scores_base, (batch_dim, base_beam_in))),
    (batch_dim, base_beam_in, 1))

  scores = scores_in + scores_base  # (batch,beam,dim)

  return scores
