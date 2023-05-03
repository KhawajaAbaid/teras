import tensorflow as tf
from teras.utils import tf_random_choice


# Soft binary gates
@tf.function
def and_operator(x, d):
    out = tf.add(x, -d + 1.5)
    out = tf.tanh(out)
    return out


@tf.function
def or_operator(x, d):
    out = tf.add(x, d - 1.5)
    out = tf.tanh(out)
    return out



# Localization
@tf.function
def broadcast_exp(x,
                  mu=None,
                  sigma=None):
    diff = tf.expand_dims(x, axis=1) - tf.expand_dims(mu, axis=0)
    loc = tf.exp(-1 * tf.norm(tf.multiply(diff, sigma), axis=-1))
    return loc


# Feature Selection
@tf.function
def binary_threshold(x, eps=0.1):
    x = tf.abs(x) - eps
    return 0.5 * binary_activation(x) + 0.5


@tf.function
def binary_activation(x):
    forward = tf.sign(x)
    backward = tf.tanh(x)
    return backward + tf.stop_gradient(forward - backward)


@tf.function
def feature_selection(input_dim,
                      keep_feature_prob_arr=None,
                      n_literals_per_formula_arr=None,
                      n_formulas=None,
                      elastic_net_beta=None,
                      learnable_mask=None,
                      elastic_net_alpha=None,
                      extension_matrix_v=None):
    binary_threshold_eps = 1
    literals_random_mask, formulas_random_mask = generate_random_mask(input_dim, keep_feature_prob_arr,
                                                                      n_literals_per_formula_arr, n_formulas)
    n_effective_features = tf.reduce_sum(formulas_random_mask, axis=0)
    formulas_random_mask = tf.reshape(formulas_random_mask, shape=tf.shape(learnable_mask))
    ext_matrix = extension_matrix(extension_matrix_v, n_formulas, n_literals_per_formula_arr)
    learnable_mask_01 = binary_threshold(learnable_mask, eps=binary_threshold_eps)

    l2_square_norm_selected = tf.linalg.diag_part(tf.matmul(tf.transpose(tf.square(learnable_mask)),
                                                            formulas_random_mask))
    l1_norm_selected = tf.linalg.diag_part(tf.matmul(tf.transpose(tf.abs(learnable_mask)),
                                                     formulas_random_mask))

    l2 = tf.abs(tf.divide(l2_square_norm_selected, n_effective_features) - elastic_net_beta * binary_threshold_eps ** 2)
    l1 = tf.abs(tf.divide(l1_norm_selected, n_effective_features) - elastic_net_beta * binary_threshold_eps)
    elastic_net_reg = tf.reduce_mean(
        (l2 * ((1 - tf.nn.sigmoid(elastic_net_alpha)) / 2) + l1 * tf.nn.sigmoid(elastic_net_alpha)))
    learnable_binary_mask = tf_matmul_sparse_dense(ext_matrix, learnable_mask_01)
    return learnable_binary_mask, literals_random_mask, elastic_net_reg


@tf.function
def generate_random_mask(input_dim,
                         keep_feature_prob_arr=None,
                         n_literals_per_formula_arr=None,
                         n_formulas=None):
    literals_random_mask = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.float32)
    formulas_random_mask = tf.TensorArray(size=n_formulas, dtype=tf.float32)

    arr_literal_index = 0
    p_index = 0
    for i in range(n_formulas):
        if i % tf.shape(n_literals_per_formula_arr)[0] == 0 and i != 0:
            p_index = (p_index + 1) % tf.shape(keep_feature_prob_arr)[0]

        p_i = tf.gather(keep_feature_prob_arr, p_index)
        mask = tf_random_choice([1., 0.], input_dim, p=[p_i, 1 - p_i])
        while tf.math.reduce_sum(mask) == 0:
            mask = tf_random_choice([1., 0.], input_dim, p=[p_i, 1 - p_i])
        n_literals_in_formula = tf.gather(n_literals_per_formula_arr, i % tf.shape(n_literals_per_formula_arr)[0])
        formulas_random_mask.write(i, tf.identity(mask))
        for _ in range(n_literals_in_formula):
            literals_random_mask.write(arr_literal_index, tf.identity(mask))
            arr_literal_index += 1

    literals_random_mask = literals_random_mask.stack()
    literals_random_mask = tf.squeeze(literals_random_mask, axis=1)
    formulas_random_mask = formulas_random_mask.stack()
    formulas_random_mask = tf.squeeze(formulas_random_mask, axis=1)
    return tf.transpose(literals_random_mask), \
           tf.transpose(formulas_random_mask)


@tf.function
def extension_matrix(v,
                     n_formulas,
                     n_literals_per_formula_arr):
    formula_index = 0
    mat = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.float32)
    arr_mat_index = 0
    for i in range(n_formulas):
        n_nodes_in_tree = tf.gather(n_literals_per_formula_arr, i % tf.shape(n_literals_per_formula_arr)[0])
        v[formula_index].assign(1.)
        for _ in range(n_nodes_in_tree):
            mat.write(arr_mat_index, tf.identity(v))
            arr_mat_index += 1
        formula_index += 1
        v[:].assign(0.)
    mat = mat.stack()
    return dense_ndim_array_to_sparse_tensor(tf.transpose(mat))


# DNF Structure
@tf.function
def create_conjunctions_indicator_matrix(b,
                                         total_number_of_conjunctions,
                                         conjunctions_depth_arr,
                                         ):
    n_different_depth = tf.shape(conjunctions_depth_arr)[0]
    n_literals_in_group = tf.reduce_sum(conjunctions_depth_arr)
    array_size = (total_number_of_conjunctions // n_different_depth) * tf.shape(conjunctions_depth_arr)[0]
    result = tf.TensorArray(size=array_size, dtype=tf.bool)
    arr_index = 0
    for i in range(total_number_of_conjunctions // n_different_depth):
        s = 0
        for d in conjunctions_depth_arr:
            b[i * n_literals_in_group + s: i * n_literals_in_group + s + d].assign(True)
            s += d
            result.write(arr_index, tf.identity(b))
            arr_index += 1
            b[:].assign(False)
    # result = tf.convert_to_tensor(result)
    result = result.stack()
    return tf.transpose(result)


@tf.function
def create_formulas_indicator_matrix(c,
                                     n_formulas,
                                     n_conjunctions_arr,
                                     ):
    result = tf.TensorArray(size=n_formulas, dtype=tf.bool)
    base = 0
    for i in range(n_formulas):
        n_conjunctions = tf.gather(n_conjunctions_arr, i % tf.shape(n_conjunctions_arr)[0])
        c[base: base + n_conjunctions].assign(True)
        result.write(i, tf.identity(c))
        c[:].assign(False)
        base += n_conjunctions
    result = result.stack()
    return tf.transpose(result)


@tf.function
def compute_total_number_of_literals(n_formulas,
                                     n_conjunctions_arr,
                                     conjunctions_depth_arr):
    return (tf.reduce_sum(n_conjunctions_arr) * tf.reduce_sum(conjunctions_depth_arr) * n_formulas) // (
                tf.shape(conjunctions_depth_arr)[0] * tf.shape(n_conjunctions_arr)[0])


@tf.function
def compute_total_number_of_conjunctions(n_formulas, n_conjunctions_arr):
    return (n_formulas // tf.shape(n_conjunctions_arr)[0]) * tf.reduce_sum(n_conjunctions_arr)


@tf.function
def compute_n_literals_per_formula(n_conjunctions_arr,
                                   conjunctions_depth_arr):
    n_literals_per_formula_arr = []
    for n_conjunctions in n_conjunctions_arr:
        n_literals_per_formula_arr.append(
            (n_conjunctions // tf.shape(conjunctions_depth_arr)[0]) * tf.reduce_sum(conjunctions_depth_arr))
    return n_literals_per_formula_arr


# General Functions
@tf.function
def dense_ndim_array_to_sparse_tensor(arr):
    # arr = arr.numpy()
    if arr.dtype == tf.bool:
        idx = tf.where(tf.not_equal(arr, False))
    else:
        idx = tf.where(tf.not_equal(arr, 0.0))
    idx = tf.cast(idx, dtype=tf.int64)
    if arr.dtype == tf.bool:
        ones_ = tf.ones(tf.shape(idx)[0])
        sh_1 = tf.cast(tf.shape(arr), dtype=tf.int64)
        return tf.SparseTensor(idx,
                               ones_,
                               sh_1
                               )
    else:
        idxx = tf.transpose(idx)
        samples = tf.gather_nd(arr, idx)
        sh_2 = tf.cast(tf.shape(arr), dtype=tf.int64)
        return tf.SparseTensor(idx,
                               samples,
                               sh_2
                               )


@tf.function
def tf_matmul_sparse_dense(sparse_A,
                           dense_B):
    return tf.transpose(tf.sparse.sparse_dense_matmul(tf.sparse.transpose(sparse_A), tf.transpose(dense_B)))

