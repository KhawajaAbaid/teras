import tensorflow as tf


@tf.function
def mask_generator(p_m, X, batch_size=None):
    """
    Generates mask vector for self and semi-supervised learning
    Args:
        p_m: corruption probability
        X: feature matrix

    Returns:
        mask: binary mask matrix

    """
    # num_samples, dim = X.shape
    # mask = np.random.binomial(1, p_m, (num_samples, dim))
    mask = tf.random.stateless_binomial(shape=tf.shape(X),
                                           seed=(0, 0),
                                           counts=1,
                                           probs=p_m,
                                           output_dtype=tf.float32)
    return mask

@tf.function
def pretext_generator(m, X, batch_size=None):
    """
    Generates corrupted samples for self and semi-supervised learning

    Args:
        m: mask matrix
        X: feature matrix
        batch_size: When X is passed from within the keras model during training, its batch dimension is none, to handle
        that particular case, pass batch_size explicity

    Returns:
        m_new: final mask matrix after corruption
        x_tilde: corrupted feature matrix
    """
    X = tf.cast(X, dtype=tf.float32)
    num_samples = tf.shape(X)[0]
    dim = tf.shape(X)[1]
    if num_samples is None:
        num_samples = batch_size

    X_bar = tf.TensorArray(size=dim, dtype=tf.float32)
    for i in range(dim):
        idx = tf.random.shuffle(tf.range(num_samples))
        X_bar = X_bar.write(i, tf.gather(X[:, i], idx))
    X_bar = tf.transpose(X_bar.stack())

    # Corrupt Samples
    X_tilde = X * (1 - m) + X_bar * m

    m_new = (X != X_tilde)
    m_new = tf.cast(m_new, dtype="float32")
    return m_new, X_tilde


def prepare_vime_dataset(X_labeled,
                         y_labeled,
                         X_unlabeled,
                         batch_size=512):
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
        X_labeled: Labeled training set
        y_labeled: Targets array for labeled training set
        X_unlabeled: Unlabeled dataset
        batch_size: Batch size
    """
    # The official implementation uses random batches at each step, we'll do the same
    num_samples_labeled = tf.cast(tf.shape(X_labeled)[0], tf.int64)
    num_samples_unlabeled = tf.cast(tf.shape(X_unlabeled)[0], tf.int64)

    X_l_ds = (tf.data.Dataset
              .from_tensor_slices({"X_labeled": X_labeled, "y_labeled": y_labeled}, name="Labeled_dataset")
              .shuffle(buffer_size=num_samples_labeled, reshuffle_each_iteration=True)
              )

    X_u_ds = (tf.data.Dataset
              .from_tensor_slices({"X_unlabeled": X_unlabeled}, "Unlabeled_dataset")
              .shuffle(buffer_size=num_samples_unlabeled, reshuffle_each_iteration=True)
              )

    # If there are fewer unlabeled samples than labeled samples, we'll repeat unlabeled dataset
    # because it is essential to have a batch of unlabeled dataset along with labeled dataset
    # for semi supervised training in VIME.
    # In case when unlabeled dataset contains more samples than labeled, we won't alter anything
    # Because by default, tensorflow dataset will produce batches equal to math.ceil(min(len(dataset_1), len(dataset_2)) / batch_size)
    if num_samples_labeled > num_samples_unlabeled:
        num_times_to_repeat = tf.cast(tf.math.ceil(20/3), dtype=tf.int64)
        X_u_ds = X_u_ds.repeat(num_times_to_repeat)

    X_merged = tf.data.Dataset.zip((X_l_ds, X_u_ds), "Merged_dataset").batch(batch_size)

    return X_merged