import numpy as np

from utils.io import load_model


def test_saved_label_encoder(config):

    encoder = load_model(
        path=config.artifacts.encoder_path
    )

    # Encoder should have learned the expected classes
    expected_classes = {
        "Dropout",
        "Enrolled",
        "Graduate",
    }

    actual_classes = set(
        encoder.classes_
    )

    assert actual_classes == expected_classes


def test_saved_label_encoder_transform(config):

    encoder = load_model(
        path=config.artifacts.encoder_path
    )

    labels = np.array([
        "Dropout",
        "Enrolled",
        "Graduate",
    ])

    encoded = encoder.transform(labels)

    assert len(encoded) == 3
    assert set(encoded) == {0, 1, 2}


def test_saved_label_encoder_inverse_transform(config):

    encoder = load_model(
        path=config.artifacts.encoder_path
    )

    encoded = np.array([0, 1, 2])

    decoded = encoder.inverse_transform(
        encoded
    )

    assert set(decoded) == {
        "Dropout",
        "Enrolled",
        "Graduate",
    }