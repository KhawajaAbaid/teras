import tensorflow as tf


@tf.function
def mask_generator(p_m, x):
    """
    Generates mask vector for self and semi-supervised learning
    Args:
        p_m: corruption probability
        x: feature matrix

    Returns:
        mask: binary mask matrix

    """
    mask = tf.random.stateless_binomial(shape=tf.shape(x),
                                           seed=(0, 0),
                                           counts=1,
                                           probs=p_m,
                                           output_dtype=tf.float32)
    return mask

@tf.function
def pretext_generator(m, x):
    """
    Generates corrupted samples for self and semi-supervised learning

    Args:
        m: mask matrix
        x: feature matrix
        batch_size: When X is passed from within the keras model during training, its batch dimension is none, to handle
        that particular case, pass batch_size explicity

    Returns:
        m_new: final mask matrix after corruption
        x_tilde: corrupted feature matrix
    """
    x = tf.cast(x, dtype=tf.float32)
    num_samples = tf.shape(x)[0]
    dim = tf.shape(x)[1]

    x_bar = tf.TensorArray(size=dim, dtype=tf.float32)
    for i in range(dim):
        idx = tf.random.shuffle(tf.range(num_samples))
        x_bar = x_bar.write(i, tf.gather(x[:, i], idx))
    x_bar = tf.transpose(x_bar.stack())

    # Corrupt Samples
    x_tilde = x * (1 - m) + x_bar * m

    m_new = (x != x_tilde)
    m_new = tf.cast(m_new, dtype="float32")
    return m_new, x_tilde


def preprocess_input_vime_semi(x_labeled,
                         y_labeled,
                         x_unlabeled,
                         batch_size=None):
    """
    VIME's semi supervised training requires a labeled and an unlabeled training set.
    It also allows user to specify the unlabeled/labeled split.
    Hence, the labeled and unlabeled dataset sizes will almost always differ.
    And here's the issue, Keras's default fit method will raise incompatible input cardinalities error
    if we try to pass labeled and unlabeled datasets to `x` parameter as a dict and pass labeled targets to `y`
    To allow for varying sizes for labeled and unlabeled dataset, this function comes to clutch.
    It fuses together the labeled and unlabeled dataset and makes sure that each labeled batch of data is accompanied
    by an unlabeled batch of data as it is required by the VIME architecture.

    Args:
        x_labeled: Labeled training set
        y_labeled: Targets array for labeled training set
        x_unlabeled: Unlabeled dataset
        batch_size: Batch size
    """
    # The official implementation uses random batches at each step, we'll do the same
    num_samples_labeled = tf.cast(tf.shape(x_labeled)[0], tf.int64)
    num_samples_unlabeled = tf.cast(tf.shape(x_unlabeled)[0], tf.int64)

    x_l_ds = (tf.data.Dataset
              .from_tensor_slices({"x_labeled": x_labeled, "y_labeled": y_labeled}, name="labeled_dataset")
              .shuffle(buffer_size=num_samples_labeled, reshuffle_each_iteration=True)
              )

    x_u_ds = (tf.data.Dataset
              .from_tensor_slices({"x_unlabeled": x_unlabeled}, "unlabeled_dataset")
              .shuffle(buffer_size=num_samples_unlabeled, reshuffle_each_iteration=True)
              )

    # If there are fewer unlabeled samples than labeled samples, we'll repeat unlabeled dataset
    # because it is essential to have a batch of unlabeled dataset along with labeled dataset
    # for semi supervised training in VIME.
    # In case when unlabeled dataset contains more samples than labeled, we won't alter anything
    # Because by default, tensorflow dataset will produce batches equal to math.ceil(min(len(dataset_1), len(dataset_2)) / batch_size)
    if num_samples_labeled > num_samples_unlabeled:
        num_times_to_repeat = tf.cast(tf.math.ceil(20/3), dtype=tf.int64)
        x_u_ds = x_u_ds.repeat(num_times_to_repeat)

    x_merged = tf.data.Dataset.zip((x_l_ds, x_u_ds), "merged_dataset")
    if batch_size is not None:
        x_merged = x_merged.batch(batch_size=batch_size)

    return x_merged


def preprocess_input_vime_self(x_unlabeled,
                               p_m=0.3,
                               m_labeled=None,
                               x_tilde=None,
                               batch_size=None
                               ):
    """
    Generates mask and corrupted samples if they aren't specified by the user
    and fuses them together with the unlabeled dataset.

    Args:
        x_unlabeled: Unlabeled dataset
        p_m: Corruption Probability
        m_labeled: Labeled mask generated by pretext_generator
        x_tilde: Corrupted dataset
        batch_size: Batch size

    Returns:
        TensorFlow dataset
    """

    if m_labeled is None and x_tilde is None:
        m_unlabeled = mask_generator(p_m,
                                     x_unlabeled)
        m_labeled, x_tilde = pretext_generator(m_unlabeled, x_unlabeled)

    dataset = tf.data.Dataset.from_tensor_slices((x_tilde,
                                                  {"mask_estimator": m_labeled, "feature_estimator": x_unlabeled}))

    if batch_size is not None:
        dataset = dataset.batch(batch_size=batch_size)

    return dataset