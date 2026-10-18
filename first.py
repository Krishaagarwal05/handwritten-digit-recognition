import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================

print("\n========================================")
print("       HANDWRITTEN DIGIT RECOGNITION")
print("========================================")

print("\nLoading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data()

print("MNIST dataset loaded successfully!")


# ============================================================
# 2. NORMALIZE DATA
# ============================================================
# Convert pixel values from 0-255 to 0-1

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# ============================================================
# 3. ADD CHANNEL DIMENSION
# ============================================================
# CNN expects:
# (images, height, width, channels)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Training data shape:", x_train.shape)
print("Testing data shape :", x_test.shape)


# ============================================================
# 4. DATA AUGMENTATION
# ============================================================
# This helps the model recognize slightly different
# handwriting styles.

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomRotation(
        0.10
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    ),

    tf.keras.layers.RandomZoom(
        height_factor=0.10,
        width_factor=0.10
    )
])


# ============================================================
# 5. CREATE CNN MODEL
# ============================================================

print("\nCreating CNN model...")

model = tf.keras.Sequential([

    # Input
    tf.keras.Input(
        shape=(28, 28, 1)
    ),

    # --------------------------------
    # Data augmentation
    # --------------------------------

    data_augmentation,


    # --------------------------------
    # Convolution Layer 1
    # --------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------
    # Convolution Layer 2
    # --------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------
    # Convolution Layer 3
    # --------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),


    # --------------------------------
    # Flatten
    # --------------------------------

    tf.keras.layers.Flatten(),


    # --------------------------------
    # Fully Connected Layer
    # --------------------------------

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),


    # --------------------------------
    # Dropout
    # --------------------------------

    tf.keras.layers.Dropout(
        0.3
    ),


    # --------------------------------
    # Output Layer
    # --------------------------------

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 6. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 7. DISPLAY MODEL
# ============================================================

print("\nCNN model created!")

model.summary()


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n========================================")
print("           TRAINING MODEL")
print("========================================")

history = model.fit(

    x_train,

    y_train,

    epochs=8,

    batch_size=64,

    validation_split=0.1
)


# ============================================================
# 9. TEST MODEL
# ============================================================

print("\n========================================")
print("          TESTING MODEL")
print("========================================")

loss, accuracy = model.evaluate(

    x_test,

    y_test,

    verbose=0
)

print(
    f"\nModel Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

model.save(
    "handwritten_cnn.keras"
)

print(
    "\nModel saved successfully!"
)


# ============================================================
# 11. LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    "handwritten_cnn.keras"
)

print(
    "Model loaded successfully!"
)


# ============================================================
# 12. FIND HANDWRITTEN IMAGES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

image_files = []

for file in os.listdir(BASE_DIR):

    if file.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):

        # Don't accidentally process
        # unrelated image files.

        image_files.append(
            os.path.join(
                BASE_DIR,
                file
            )
        )


# Sort files alphabetically

image_files.sort()


print("\n========================================")
print("      CHECKING FOR HANDWRITTEN IMAGES")
print("========================================")


if len(image_files) == 0:

    print(
        "\nNO IMAGES FOUND!"
    )

    print(
        "\nPut your images in:"
    )

    print(
        BASE_DIR
    )

    print(
        "\nExample:"
    )

    print(
        "digit1.png"
    )

    print(
        "digit2.png"
    )

else:

    print(
        f"\nFound {len(image_files)} image(s):"
    )

    for image_path in image_files:

        print(
            " -",
            os.path.basename(image_path)
        )


# ============================================================
# 13. PROCESS EACH IMAGE
# ============================================================

for image_path in image_files:

    print("\n")
    print("========================================")

    print(
        "Processing:",
        os.path.basename(image_path)
    )

    print(
        "========================================")


    try:

        # ====================================================
        # STEP 1: READ IMAGE
        # ====================================================

        original = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if original is None:

            print(
                "ERROR: Could not read image."
            )

            continue


        print(
            "Original image size:",
            original.shape
        )


        # ====================================================
        # STEP 2: THRESHOLD
        # ====================================================
        # Converts image to black and white.
        #
        # Black background -> 0
        # White digit     -> 255

        _, thresh = cv2.threshold(

            original,

            0,

            255,

            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )


        # ====================================================
        # STEP 3: REMOVE SMALL NOISE
        # ====================================================

        kernel = np.ones(
            (2, 2),
            np.uint8
        )

        thresh = cv2.morphologyEx(

            thresh,

            cv2.MORPH_OPEN,

            kernel
        )


        # ====================================================
        # STEP 4: FIND NON-BLACK PIXELS
        # ====================================================

        coords = cv2.findNonZero(
            thresh
        )


        if coords is None:

            print(
                "ERROR: No digit detected!"
            )

            continue


        # ====================================================
        # STEP 5: FIND DIGIT BOUNDING BOX
        # ====================================================

        x, y, w, h = cv2.boundingRect(
            coords
        )


        print(
            f"Detected digit area: "
            f"{w} x {h}"
        )


        # ====================================================
        # STEP 6: CROP DIGIT
        # ====================================================

        digit = thresh[
            y:y+h,
            x:x+w
        ]


        # ====================================================
        # STEP 7: MAKE DIGIT SQUARE
        # ====================================================

        height, width = digit.shape

        size = max(
            height,
            width
        )


        square = np.zeros(

            (size, size),

            dtype=np.uint8
        )


        # Center digit horizontally

        x_offset = (
            size - width
        ) // 2


        # Center digit vertically

        y_offset = (
            size - height
        ) // 2


        square[

            y_offset:
            y_offset + height,

            x_offset:
            x_offset + width

        ] = digit


        # ====================================================
        # STEP 8: RESIZE TO 20 x 20
        # ====================================================

        digit = cv2.resize(

            square,

            (20, 20),

            interpolation=cv2.INTER_AREA
        )


        # ====================================================
        # STEP 9: CREATE 28 x 28 MNIST IMAGE
        # ====================================================

        final_image = np.zeros(

            (28, 28),

            dtype=np.uint8
        )


        # Put 20x20 digit in center

        final_image[

            4:24,

            4:24

        ] = digit


        # ====================================================
        # STEP 10: NORMALIZE
        # ====================================================

        final_image = (

            final_image.astype(
                "float32"
            ) / 255.0

        )


        # ====================================================
        # STEP 11: PREPARE IMAGE FOR CNN
        # ====================================================

        input_image = final_image[
            np.newaxis,
            ...,
            np.newaxis
        ]


        print(
            "Final image shape:",
            input_image.shape
        )


        # ====================================================
        # STEP 12: PREDICT
        # ====================================================

        prediction = model.predict(

            input_image,

            verbose=0
        )


        # ====================================================
        # STEP 13: GET PREDICTED DIGIT
        # ====================================================

        predicted_digit = np.argmax(
            prediction[0]
        )


        # ====================================================
        # STEP 14: GET CONFIDENCE
        # ====================================================

        confidence = (

            np.max(
                prediction[0]
            ) * 100

        )


        # ====================================================
        # STEP 15: DISPLAY ALL PROBABILITIES
        # ====================================================

        print(
            "\nPrediction probabilities:"
        )

        print(
            "----------------------------"
        )


        for i in range(10):

            probability = (

                prediction[0][i]
                * 100

            )

            print(

                f"Digit {i}: "
                f"{probability:.2f}%"

            )


        # ====================================================
        # STEP 16: FINAL RESULT
        # ====================================================

        print(
            "\n****************************"
        )

        print(
            f"PREDICTED DIGIT: "
            f"{predicted_digit}"
        )

        print(
            f"CONFIDENCE: "
            f"{confidence:.2f}%"
        )

        print(
            "****************************"
        )


        # ====================================================
        # STEP 17: DISPLAY PROCESSED IMAGE
        # ====================================================

        plt.figure(
            figsize=(5, 5)
        )


        plt.imshow(

            final_image,

            cmap="gray"
        )


        plt.title(

            f"Predicted Digit: "
            f"{predicted_digit}\n"
            f"Confidence: "
            f"{confidence:.2f}%"

        )


        plt.axis("off")


        plt.show()


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            "\nERROR WHILE PROCESSING IMAGE:"
        )

        print(
            str(e)
        )


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("       FINISHED PROCESSING IMAGES")
print("========================================")

