import tensorflow as tf
from teras.layers.dnfnet import FeatureSelection, Localization, DNNF
from teras.utils.dnfnet import compute_num_literals_per_formula


def test_feature_selection_output_shape():
    feature_selection = FeatureSelection(input_dim=16,
                                         num_formulas=32)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    num_conjunctions_arr = [6, 9, 12, 15]
    conjunctions_depth_arr = [2, 4, 6]
    num_literals_per_formula_arr = compute_num_literals_per_formula(num_conjunctions_arr,
                                                                    conjunctions_depth_arr)

    learnable_binary_mask, literals_random_mask, elastic_net_reg = feature_selection(num_literals_per_formula_arr)
    print(tf.shape(learnable_binary_mask))
    print(tf.shape(literals_random_mask))
    print(tf.shape(elastic_net_reg))
    assert len(tf.shape(learnable_binary_mask)) == 2
    assert tf.shape(learnable_binary_mask)[0] == 16
    assert tf.shape(learnable_binary_mask)[1] == 0
    assert len(tf.shape(literals_random_mask)) == 2
    assert tf.shape(literals_random_mask)[0] == 16
    assert tf.shape(literals_random_mask)[1] == 0
    assert len(tf.shape(elastic_net_reg)) == 0


def test_localitzation_output_shape():
    localization = Localization(num_formulas=32)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = localization(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 32


def test_dnnf_output_shape():
    dnnf = DNNF(num_formulas=32)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = dnnf(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 32
