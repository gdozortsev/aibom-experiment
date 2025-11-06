import keras
from keras import layers
import os
import tensorflow as tf
import keras.backend as K
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import random

@keras.saving.register_keras_serializable(name="CombinedCEAndDiceLoss")
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha

    def dice_loss(self, y_true, y_pred):
        smooth = K.epsilon()

        # Convert y_true to one-hot format with depth 3 (0, 1, 2)
        # use `tf.one_hot` after using `tf.cast` to convert y_true to int32
        # the `tf.one_hot` depth should be 3 for 3 classes
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)

        # Flatten the tensors
        # use `tf.reshape` to flatten both y_true_one_hot and y_pred to shape (-1, <num_classes>)
        y_true_f = tf.reshape(y_true_one_hot, (-1, 3))
        y_pred_f = tf.reshape(y_pred, (-1, 3))

        # Calculate Dice coefficient for each class
        # Intersection is the element-wise multiplication (`*`) of y_true_f and y_pred_f
        # use `tf.reduce_sum` to sum the intersection over all pixels
        # remember to use the `axis=0` argument to sum over all pixels
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        # Union is the element-wise sum (`+`) of true and predicted pixels for each class
        # again use `tf.reduce_sum` to sum the union over all pixels (remember `axis=0`)
        union = tf.reduce_sum(y_true_f + y_pred_f, axis=0)

        # Calculate Dice coefficient for each class:
        # (2 * intersection + smooth) / (union + smooth)
        dice = (2 * intersection + smooth) / (union + smooth)

        # Dice loss for multi-class, averaged over all classes
        # remember to use 1 - mean(dice) as the dice loss
        # use `tf.reduce_mean` on `dice` to average over all classes
        dice_loss = 1 - tf.reduce_mean(dice)

        return dice_loss

    def ce_loss(self, y_true, y_pred):
        # y_true is a 4D int8 tensor with shape (batch_size, height, width, 1)
        # y_pred is a 4D int8 tensor with shape (batch_size, height, width, num_classes)
        # use `keras.losses.SparseCategoricalCrossentropy()` loss with y_true and y_pred
        cross_entropy = keras.losses.SparseCategoricalCrossentropy()
        return cross_entropy(y_true, y_pred)

    def call(self, y_true, y_pred):
        # Calculate the combined loss
        # use self.alpha to weight the `self.dice_loss()` and `self.ce_loss()` losses.
        # call the `self.dice_loss()` and `self.ce_loss()` functions with y_true and y_pred.
        # return the combined loss
        return self.alpha * self.dice_loss(y_true, y_pred) + (1 - self.alpha) * self.ce_loss(y_true, y_pred)
def main():
    #TODO change these directories during build
    input_dir = "/Users/gdozorts/Downloads/AIBOM/application/images"
    target_dir = "/Users/gdozorts/Downloads/AIBOM/application/annotations/trimaps"

    img_size = (160, 160)
    num_classes = 3
    batch_size = 32

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    """build a TF Dataset to easily read the images and annotations"""


    def get_dataset(
        batch_size,
        img_size,
        input_img_paths,
        target_img_paths,
        max_dataset_len=None,
    ):
        """Returns a TF Dataset."""

        def load_img_masks(input_img_path, target_img_path):
            input_img = tf_io.read_file(input_img_path)
            input_img = tf_io.decode_png(input_img, channels=3)
            input_img = tf_image.resize(input_img, img_size)
            input_img = tf_image.convert_image_dtype(input_img, "float32")

            target_img = tf_io.read_file(target_img_path)
            target_img = tf_io.decode_png(target_img, channels=1)
            target_img = tf_image.resize(target_img, img_size, method="nearest")
            target_img = tf_image.convert_image_dtype(target_img, "uint8")

            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            target_img -= 1
            return input_img, target_img

        # For faster debugging, limit the size of data
        if max_dataset_len:
            input_img_paths = input_img_paths[:max_dataset_len]
            target_img_paths = target_img_paths[:max_dataset_len]
        dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
        dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
        return dataset.batch(batch_size)

    """### Model Implementation

    A U-NET model which consists of:
    - a convolutional Encoder with 3 layers where each has stacked 2D seperable convolutions (they have less weights).
    -  a residual connection 
        - taking the output of the last layer, upsampling it, and adding it back to the output of the current layer.
    [Ronnenberger et al](https://arxiv.org/pdf/1505.04597.pdf).
    """


    def get_unet_model(img_size, num_classes):
        '''
        Builds the UNet model with MobileNetV2 encoder.
        Args:
            img_size: The input image size.
            num_classes: The number of output classes.

        Uses the following layers from keras:
            - keras.layers.Input
            - keras.layers.Conv2D
            - keras.layers.BatchNormalization
            - keras.layers.Activation
            - keras.layers.SeparableConv2D
            - keras.layers.MaxPooling2D
            - keras.layers.UpSampling2D
            - keras.layers.Concatenate
            - keras.layers.Conv2DTranspose

        Returns:
            The UNet model.
        '''

        inputs = keras.Input(shape=img_size + (3,))

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        # add a conv2d layer with 32 filters, 3x3 kernel size, padding "same", and stride of 2
        # use `layers.Conv2D`
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        # add batch normalization using `layers.BatchNormalization`
        x = layers.BatchNormalization()(x)
        # add a relu activation using `layers.Activation` with "relu"
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        skip_connections = [x]

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            # add layers.Activation on top of previous block with "relu"
            x = layers.Activation("relu")(x)
            # add a layers.SeparableConv2D layer with filters, 3x3 kernel size, and padding "same"
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            # add batch normalization
            x = layers.BatchNormalization()(x)


            # add activation on top of previous block with "relu"
            x = layers.Activation("relu")(x)
            # add a layers.SeparableConv2D layer with filters, 3x3 kernel size, and padding "same"
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            # add batch normalization
            x = layers.BatchNormalization()(x)

            # save the outputs to be used in skip connections
            skip_connections.append(x)

            # add a layers.MaxPooling2D layer with pool size 3 and stride 2 and padding "same"
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            # add a layers.Conv2D layer with filters, 1x1 kernel size, and stride of 2, padding "same"
            # on top of previous_block_activation
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
            # Add back `residual` to `x` with `layers.add()`
            x = layers.add([x, residual])  # Add back residual

            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for i, filters in enumerate( [256, 128, 64, 32]):
            if i > 0:
                # add `skip_connections[-i]` as input to this block
                # with `layers.Concatenate()` add the skip connection to `x`
                x = layers.Concatenate()([x, skip_connections[-i]])

            # add layers.Activation on top of previous block, with "relu"
            x = layers.Activation("relu")(x)
            # add a layers.Conv2DTranspose layer with filters, 3x3 kernel size, and padding "same"
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            # add batch normalization
            x = layers.BatchNormalization()(x)

            # add layers.Activation on top of previous block, with "relu"
            x = layers.Activation("relu")(x)
            # add a layers.Conv2DTranspose layer with filters, 3x3 kernel size, and padding "same"
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            # add batch normalization
            x = layers.BatchNormalization()(x)

            # add layers.UpSampling2D layer with size 2
            x = layers.UpSampling2D(2)(x)

            # Project residual
            # upsample (`layers.UpSampling2D`) the `previous_block_activation` with factor 2
            residual = layers.UpSampling2D(2)(previous_block_activation)
            # add a layers.Conv2D with filters, 1x1 kernel size, and padding "same"
            residual = layers.Conv2D(filters, 1, padding="same")(residual)

            # Add back residual to x like before with `layers.add()`
            x = layers.add([x, residual])

            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        # use a layers.Conv2D with num_classes filters, 3x3 kernel size, and padding "same"
        # with "softmax" activation, and num_classes as output depth (e.g. numebr of kernels)
        outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model


    # Build model
    model = get_unet_model(img_size, num_classes)

    """### Prepare data for training"""

    # Split our img paths into a training and a validation set
    val_samples = 1000
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate dataset for each split
    # Limit input files in `max_dataset_len` for faster epoch training time.
    # Remove the `max_dataset_len` arg when running with full dataset.
    train_dataset = get_dataset(
        batch_size,
        img_size,
        train_input_img_paths,
        train_target_img_paths,
        max_dataset_len=1000,
    )
    valid_dataset = get_dataset(
        batch_size, img_size, val_input_img_paths, val_target_img_paths
    )




    """### Model training

    Compile the model and run training for 1 epochs.
    """

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=CombinedLoss())
    callbacks = [
        keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)
    ]


    # Train the model, doing validation at the end of each epoch.
    # TODO Originally epochs was set to 30, but I set it to 1 to run faster for testing
    epochs = 1
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        verbose=2,
    )
if __name__ == "__main__":
    main()