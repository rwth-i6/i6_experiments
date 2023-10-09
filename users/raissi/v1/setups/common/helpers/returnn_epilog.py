def get_epilog_code_dense_label(
    n_input,
    n_contexts,
    n_states,
    use_word_end_classes=False,
    is_min_duration=False,
    specaugment=None,
):
    # Remember here you need to distinguish between the label calculation for state-tying in rasr
    # and the possibility of having one model for all 3 states. Here, the first aspect
    # is done by setting nStates and the second with isMinDuration
    common = (
        """
from TFUtil import DimensionTag
time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="time")

extern_data = {'data': {"dim": %d, "same_dim_tags_as": {"t": time_tag}}}
    """
        % n_input
    )

    if use_word_end_classes:
        n_classes = n_contexts**3 * n_states * 2
        n_center_state_classes = n_contexts * 2 if is_min_duration else n_contexts * n_states * 2

    else:
        n_classes = n_contexts**3 * n_states
        n_center_state_classes = n_contexts if is_min_duration else n_contexts * n_states

    graph_code = """

numbers = [%d, %d, %d, %d]
for i, k in enumerate(['classes', 'centerState', 'pastLabel', 'futureLabel']):
  extern_data[k] = {'dim': numbers[i], 'dtype': 'int32', 'sparse': True, "same_dim_tags_as": {"t": time_tag},  "available_for_inference": True}
                 """ % (
        n_classes,
        n_center_state_classes,
        n_contexts,
        n_contexts,
    )

    finalCode = common + graph_code
    if specaugment is not None:
        finalCode += specaugment

    return finalCode
