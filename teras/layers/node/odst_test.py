import tensorflow as tf
from teras.layers.node.odst import ObliviousDecisionTree


def test_node_oblivious_decision_tree_valid_call():
    oblivious = ObliviousDecisionTree(num_trees=16,
                                      depth=8,
                                      tree_dim=4)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = oblivious(inputs)


def test_node_oblivious_decision_tree_output_shape():
    oblivious = ObliviousDecisionTree(num_trees=16,
                                      depth=8,
                                      tree_dim=4)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = oblivious(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16 * 4   # num_tress * tree_dim
