import tensorflow as tf
from teras.layers.vime.vime_mask_generation_and_corruption import VimeMaskGenerationAndCorruption


def test_vime_feature_estimator_valid_call():
    inputs = tf.ones((8, 5))
    gen_mask_and_corrupt = VimeMaskGenerationAndCorruption(p_m=0.3)
    outputs = gen_mask_and_corrupt(inputs)


def test_vime_feature_estimator_output_shape():
    inputs = tf.ones((8, 5))
    gen_mask_and_corrupt = VimeMaskGenerationAndCorruption(p_m=0.3)
    outputs = gen_mask_and_corrupt(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 5
