"""
FlowGuard Penang: TensorFlow MobileNetV2 Transfer Learning Pipeline.

This script documents and executes the official Slide 4 & Slide 5 training
architecture:
- Base backbone: MobileNetV2 (ImageNet weights, 224x224x3 input, include_top=False)
- Week 10 Augmentation Layer: RandomBrightness(0.25) + RandomContrast(0.25)
  plus RandomFlip("horizontal") to eliminate false alarms from afternoon shadows
- Classification Head: GlobalAveragePooling2D -> Dropout(0.25) -> Dense(3, Softmax)
- Automatic fallback to calibrated portable weight training if TensorFlow is
  not installed in the current environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def build_mobilenetv2_keras_model(num_classes: int = 3):
    """
    Construct the TensorFlow MobileNetV2 transfer learning architecture.

    Selected over heavier models (ResNet50, EfficientNetB4) because MobileNetV2
    uses depthwise-separable convolutions (~3.5M parameters), enabling low-latency
    CPU inference on standard laptops and municipal field devices.
    """
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore

    # Week 10 Alpha Bug Fix: Brightness and contrast augmentation during training
    # prevents the classifier from associating dark tree shadows with trash.
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomBrightness(factor=0.25),
            layers.RandomContrast(factor=0.25),
        ],
        name="week10_shadow_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze early convolutional edge detectors; fine-tune top semantic blocks
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="drain_rgb_224")
    x = data_augmentation(inputs)
    # MobileNetV2 expects pixel intensities scaled to [-1.0, 1.0]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.25, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="severity_softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="FlowGuard_Penang_MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_tensorflow_pipeline(dataset_dir: Path, epochs: int = 12, batch_size: int = 16) -> Tuple[float, Path]:
    """Train the Keras MobileNetV2 model on labeled Clear/Moderate/Critical folders."""
    import tensorflow as tf  # type: ignore

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train",
        label_mode="categorical",
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "val",
        label_mode="categorical",
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    model = build_mobilenetv2_keras_model(num_classes=3)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    best_val_acc = float(max(history.history.get("val_accuracy", [0.0]))) * 100.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "flowguard_mobilenetv2.keras"
    model.save(str(out_path))
    return round(best_val_acc, 1), out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlowGuard Penang blockage classifier.")
    parser.add_argument("--dataset", type=str, default=str(ROOT / "dataset"), help="Path with train/ and val/ folders.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of fine-tuning epochs.")
    args = parser.parse_args()

    try:
        import tensorflow as tf  # type: ignore  # noqa: F401
        has_tf = True
    except ImportError:
        has_tf = False

    ds_path = Path(args.dataset)
    if has_tf and (ds_path / "train").exists() and (ds_path / "val").exists():
        print("TensorFlow detected. Starting MobileNetV2 transfer learning...")
        val_acc, model_path = train_tensorflow_pipeline(ds_path, epochs=args.epochs)
        print(f"Saved Keras model to {model_path} (val_accuracy={val_acc}%)")
    else:
        print("Running portable calibration pipeline via generate_sample_dataset.py...")
        from generate_sample_dataset import main as run_portable_training

        run_portable_training()


if __name__ == "__main__":
    main()
