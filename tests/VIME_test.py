from tensorflow.keras.datasets import mnist
from tensorflow import keras
from sklearn.model_selection import train_test_split
from teras.models import VimeSemi, VimeSelf
from teras.utils import prepare_vime_dataset
import tensorflow as tf
import pandas as pd
import numpy as np
import os




def vime_main(labeled_data_ratio=0.5,
              p_m=0.3,
              alpha=2.0,
              beta=1.0,
              K=3):
    """
    Args:
        model_sets: supervised model sets
        num_labeled_samples: number of labeled samples/data to be used
        p_m: corruption probability
        alpha: hyper-parameter to control two self-supervied loss
        K: number of augmented data
        beta: hyper-parameter to control two semi-supervied loss
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # One hot encoding for the labels
    y_train = np.asarray(pd.get_dummies(y_train))
    y_test = np.asarray(pd.get_dummies(y_test))

    # Normalize features
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape
    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1] * X_train.shape[2]])
    X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1] * X_test.shape[2]])
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, y_train,
                                                            test_size=labeled_data_ratio,
                                                            shuffle=True)

    X_unlabeled = X_unlabeled[:10000]
    X_labeled_train, X_labeled_val, y_labeled_train, y_labeled_val = train_test_split(X_labeled, y_labeled,
                                                                                      test_size=0.1,
                                                                                      shuffle=True)

    X_final_ds = prepare_vime_dataset(X_labeled_train, y_labeled_train, X_unlabeled, 1024)


    # Train vime self
    print("Training VIME self supervised part...")
    v_self = VimeSelf(p_m, alpha)
    v_self.fit(X_unlabeled, batch_size=128, epochs=10)
    trained_encoder = v_self.get_encoder()
    encoder_file_path = "./vime_saves/encoder/"
    if not os.path.exists(encoder_file_path):
        os.makedirs(encoder_file_path)
    trained_encoder.save(encoder_file_path + "encoder_mode.h5")

    # Train vime semi
    print("\n\nTraining VIME semi supervised part...")

    v_semi = VimeSemi(hidden_dim=128,
                      input_dim=784,
                      encoder_file_path="./vime_saves/encoder/encoder_mode.h5",
                      num_labels=10,
                      batch_size=1024)
    v_semi.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                   metrics=[keras.metrics.CategoricalAccuracy()])
    v_semi.fit(X_final_ds, epochs=5)

    print("\n\nEvaluating...")
    v_semi.evaluate(X_labeled_val, y_labeled_val)


if __name__ == "__main__":
    vime_main()
