import tensorflow as tf
from teras.layers.dnfnet.dnfnet_feature_selection import DNFNetFeatureSelection
from teras.utils.dnfnet import compute_num_literals_per_formula


def test_dnfnet_feature_selection_valid_call():
    feature_selection = DNFNetFeatureSelection(input_dim=10,
                                               num_formulas=16)
    inputs = tf.ones((8, 10), dtype=tf.float32)
    num_conjunctions_arr = [6, 9, 12, 15]
    conjunctions_depth_arr = [2, 4, 6]
    num_literals_per_formula_arr = compute_num_literals_per_formula(num_conjunctions_arr,
                                                                    conjunctions_depth_arr)

    learnable_binary_mask, literals_random_mask, elastic_net_reg = feature_selection(num_literals_per_formula_arr)


def test_dnfnet_feature_selection_output_shape():
    feature_selection = DNFNetFeatureSelection(input_dim=10,
                                               num_formulas=16)
    inputs = tf.ones((8, 10), dtype=tf.float32)
    num_conjunctions_arr = [6, 9, 12, 15]
    conjunctions_depth_arr = [2, 4, 6]
    num_literals_per_formula_arr = compute_num_literals_per_formula(num_conjunctions_arr,
                                                                    conjunctions_depth_arr)

    learnable_binary_mask, literals_random_mask, elastic_net_reg = feature_selection(num_literals_per_formula_arr)
    assert len(tf.shape(learnable_binary_mask)) == 2
    assert tf.shape(learnable_binary_mask)[0] == 10
    assert tf.shape(learnable_binary_mask)[1] == 0
    assert len(tf.shape(literals_random_mask)) == 2
    assert tf.shape(literals_random_mask)[0] == 10
    assert tf.shape(literals_random_mask)[1] == 0
    assert len(tf.shape(elastic_net_reg)) == 0
