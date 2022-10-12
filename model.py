import ee
import folium
import numpy as np
import tensorflow as tf
from pprint import pprint
from sklearn.metrics import mean_squared_error

def parse_tfrecord(example_proto):
    """The parsing function.

    Read a serialized example into the structure defined by FEATURES_DICT.

    Args:
      example_proto: a serialized Example.

    Returns:
      A tuple of the predictors dictionary and the label, cast to an `int32`.
    """
    parsed_features = tf.io.parse_single_example(example_proto, FEATURES_DICT)
    labels = parsed_features.pop(LABEL)
    return parsed_features, tf.cast(labels, tf.int32)


def to_tuple(inputs, label):
    """Convert inputs to a tuple.

    Note that the inputs must be a tuple of tensors in the right shape.

    Args:
      dict: a dictionary of tensors keyed by input name.
      label: a tensor storing the response variable.

    Returns:
      A tuple of tensors: (predictors, label).
    """
    # Values in the tensor are ordered by the list of predictors.
    predictors = [inputs.get(k) for k in BANDS]
    return (
        tf.expand_dims(tf.transpose(predictors), 1),
        tf.expand_dims(tf.expand_dims(label, 1), 1),
    )

def landuse(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    landuse_mask = forest
    # Both flags should be set to zero, indicating clear conditions.
    return image.updateMask(landuse_mask)

DATA_BUCKET = "rabpro-gee-uploads"
file_extension = ".tfrecord.gz"
TRAIN_FILE_PREFIX = "logistic_demo_training"
TRAIN_FILE_PATH = (
    "gs://" + DATA_BUCKET + "/ps_points/" + TRAIN_FILE_PREFIX + file_extension
)
TEST_FILE_PREFIX = "logistic_demo_testing"
TEST_FILE_PATH = (
    "gs://" + DATA_BUCKET + "/ps_points/" + TEST_FILE_PREFIX + file_extension
)

OPTICAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
NDVI_BANDS = ["NDVI"]
NDWI = ["NDWI"]
BANDS = [NDVI_BANDS + NDWI]
FEATURE_NAMES = list(BANDS)
LABEL = "loss19"
FEATURE_NAMES.append(LABEL)
columns = [tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in FEATURE_NAMES]
FEATURES_DICT = dict(zip(FEATURE_NAMES, columns))

OUTPUT_BUCKET = "rabpro-gee-uploads"
MODEL_DIR = "gs://" + OUTPUT_BUCKET + "/logistic_model"
MODEL_NAME = "logistic_model5"
VERSION_NAME = "v0"

# + colab={"base_uri": "https://localhost:8080/"} id="qTFY279qJk4q" outputId="6089310a-cf07-4d1d-dde9-bc99002000d7"
# Load datasets from the files.
train_dataset = tf.data.TFRecordDataset(TRAIN_FILE_PATH, compression_type="GZIP")
test_dataset = tf.data.TFRecordDataset(TEST_FILE_PATH, compression_type="GZIP")

export_image = "projects/jsta-pspoints/assets/logistic_demo_imageNDWI"
stack = ee.Image(export_image)

# Compute the size of the shuffle buffer.  We can get away with this
# because it's a small dataset, but watch out with larger datasets.
train_size = 0
for _ in iter(train_dataset):
    train_size += 1

batch_size = 8

# Map the functions over the datasets to parse and convert to tuples.
train_dataset = train_dataset.map(parse_tfrecord, num_parallel_calls=4)
train_dataset = train_dataset.map(to_tuple, num_parallel_calls=4)
train_dataset = train_dataset.shuffle(train_size).batch(batch_size)

test_dataset = test_dataset.map(parse_tfrecord, num_parallel_calls=4)
test_dataset = test_dataset.map(to_tuple, num_parallel_calls=4)
test_dataset = test_dataset.batch(batch_size)

# Print the first parsed record to check.
pprint(iter(train_dataset).next())

# + colab={"base_uri": "https://localhost:8080/"} id="_X5I6Zy6Jnsj" outputId="09738b0e-80a8-4242-fe62-9af79797b9bf"

# model = tf.keras.models.Sequential(RandomForestModel(num_trees=30))

# Define the layers in the model.
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input((1, 1, len(BANDS))),
        tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid"),
        tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid"),
        tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid"),
    ]
)


# Compile the model with the specified loss function.
model.compile(
    optimizer=tf.keras.optimizers.SGD(momentum=0.9),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Fit the model to the training data.
model.fit(x=train_dataset, epochs=20, validation_data=test_dataset)

# + colab={"base_uri": "https://localhost:8080/"} id="wn2cMW39VMth" outputId="e94c72b6-dcd4-46ab-b658-554e0b230413"
model.save(MODEL_DIR, save_format="tf")

# + colab={"base_uri": "https://localhost:8080/"} id="Mt--6MCHVaJu" outputId="259b9bd7-547c-4e22-9c29-3db6dbf5cdfd"
from tensorflow.python.tools import saved_model_utils

meta_graph_def = saved_model_utils.get_meta_graph_def(MODEL_DIR, "serve")
inputs = meta_graph_def.signature_def["serving_default"].inputs
outputs = meta_graph_def.signature_def["serving_default"].outputs

# Just get the first thing(s) from the serving signature def.  i.e. this
# model only has a single input and a single output.
input_name = None
for k, v in inputs.items():
    input_name = v.name
    break

output_name = None
for k, v in outputs.items():
    output_name = v.name
    break

# Make a dictionary that maps Earth Engine outputs and inputs to
# AI Platform inputs and outputs, respectively.
import json

input_dict = "'" + json.dumps({input_name: "array"}) + "'"
output_dict = "'" + json.dumps({output_name: "output"}) + "'"
print(input_dict)
print(output_dict)

# + id="4VZow-zHVmA3"
PROJECT = "quick-keel-352020"
EEIFIED_DIR = "gs://" + OUTPUT_BUCKET + "/logistic_eeified"

# + colab={"base_uri": "https://localhost:8080/"} id="dFIwfUBqVoRq" outputId="e8b65972-35c7-4a7e-c6de-c8345e243743"
# !earthengine set_project {PROJECT}
# !earthengine model prepare --source_dir {MODEL_DIR} --dest_dir {EEIFIED_DIR} --input {input_dict} --output {output_dict}

# + colab={"base_uri": "https://localhost:8080/"} id="eEAOenuDVyHa" outputId="3e8fa9cd-acb4-49be-daae-1d44fa1c6769"
# !gcloud ai-platform models create {MODEL_NAME} \
#   --project {PROJECT} \
#   --region {REGION}

# !gcloud ai-platform versions create {VERSION_NAME} \
#   --project {PROJECT} \
#   --region {REGION} \
#   --model {MODEL_NAME} \
#   --origin {EEIFIED_DIR} \
#   --framework "TENSORFLOW" \
#   --runtime-version=2.3 \
#   --python-version=3.7

# + colab={"base_uri": "https://localhost:8080/", "height": 712} id="v-IjY8nXaysO" outputId="6e0f1db0-9e16-48ef-e384-fc4b4a1c8a6d"
# Turn into an array image for input to the model.
array_image = stack.select(BANDS).float().toArray()

# Point to the model hosted on AI Platform.  If you specified a region other
# than the default (us-central1) at model creation, specify it here.
model = ee.Model.fromAiPlatformPredictor(
    projectName=PROJECT,
    modelName=MODEL_NAME,
    version=VERSION_NAME,
    # Can be anything, but don't make it too big.
    inputTileSize=[8, 8],
    # Keep this the same as your training data.
    proj=ee.Projection("EPSG:4326").atScale(30),
    fixInputProj=True,
    # Note the names here need to match what you specified in the
    # output dictionary you passed to the EEifier.
    outputBands={"output": {"type": ee.PixelType.float(), "dimensions": 1}},
)

# Output probability.
predictions = model.predictImage(array_image).arrayGet([0])

# Back-of-the-envelope decision rule.
predicted = predictions.gt(0.5).selfMask()

# Training data for comparison.
IMAGE = ee.Image("projects/ee-pwashaya9/assets/tezba_raster_binary2")
LOSS_image = ee.ImageCollection([IMAGE]).map(landuse)
LOSS_19 = (((IMAGE).select("b1")).rename(LABEL)).uint8()
LOSS19 = LOSS_19.gt(0.5)
reference = LOSS19.selfMask()

# Get map IDs for display in folium.
probability_vis = {"min": 0, "max": 1}
probability_mapid = predictions.getMapId(probability_vis)

predicted_vis = {"palette": "red"}
predicted_mapid = predicted.getMapId(predicted_vis)

reference_vis = {"palette": "orange"}
reference_mapid = reference.getMapId(reference_vis)

image_vis = {"bands": ["NDVI"], "min": 0, "max": 0.3}
image_mapid = NDVI_2019_.getMapId(image_vis)

# Visualize the input imagery and the predictions.
map = folium.Map(location=[-9.1, -62.3], zoom_start=11)
folium.TileLayer(
    tiles=image_mapid["tile_fetcher"].url_format,
    attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    overlay=True,
    name="image",
).add_to(map)
folium.TileLayer(
    tiles=probability_mapid["tile_fetcher"].url_format,
    attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    overlay=True,
    name="probability",
).add_to(map)
folium.TileLayer(
    tiles=predicted_mapid["tile_fetcher"].url_format,
    attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    overlay=True,
    name="predicted",
).add_to(map)
folium.TileLayer(
    tiles=reference_mapid["tile_fetcher"].url_format,
    attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    overlay=True,
    name="reference",
).add_to(map)
map.add_child(folium.LayerControl())
map

# + id="SuSB8aeRiNRQ"
y_test = LOSS19.toArray()

# + colab={"base_uri": "https://localhost:8080/"} id="Ra4qnmUJjv5M" outputId="b85ba1e4-4faa-4726-97b8-3b3ccb0a3f7e"
print(y_test)

# + colab={"base_uri": "https://localhost:8080/", "height": 433} id="jJiSAp7ahQ7W" outputId="294fb4aa-91ef-4416-a31a-82b92b8991e2"
np.sqrt(mean_squared_error(test_dataset, predictions))
