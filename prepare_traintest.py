import ee
import tensorflow as tf


PROJECT = "jsta-pspoints"
OUTPUT_BUCKET = "rabpro-gee-uploads"
DATA_BUCKET = "rabpro-gee-uploads"
REGION = "us-central1"
TRAIN_FILE_PREFIX = "logistic_demo_training"
TEST_FILE_PREFIX = "logistic_demo_testing"
file_extension = ".tfrecord.gz"
TRAIN_FILE_PATH = (
    "gs://" + DATA_BUCKET + "/ps_points/" + TRAIN_FILE_PREFIX + file_extension
)
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

MODEL_DIR = "gs://" + OUTPUT_BUCKET + "/logistic_model"
MODEL_NAME = "logistic_model5"
VERSION_NAME = "v0"


def landuse(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    landuse_mask = forest
    # Both flags should be set to zero, indicating clear conditions.
    return image.updateMask(landuse_mask)

GEOMETRY = ee.Geometry.Polygon(
    [
        [
            [11.891839177291098, 48.51063424579321],
            [19.0768977710411, 48.51063424579321],
            [19.0768977710411, 51.22876410463182],
            [11.891839177291098, 51.22876410463182],
            [11.891839177291098, 48.51063424579321],
        ]
    ],
    None,
    False,
)

corine = ee.Image("COPERNICUS/CORINE/V18_5_1/100m/2012").clip(GEOMETRY)
forest = corine.updateMask(corine.gte(22).And(corine.lte(25)))

# Forest loss in 2016 is what we want to predict.
IMAGE = ee.Image("projects/ee-pwashaya9/assets/tezba_raster_binary2")
LOSS_image = ee.ImageCollection([IMAGE]).map(landuse)
LOSS_19 = (((IMAGE).select("b1")).rename(LABEL)).uint8()
LOSS19 = LOSS_19.gt(0.5)

export_image = "projects/jsta-pspoints/assets/logistic_demo_imageNDWI"

# + id="zNw8hjOtJcAz"
sample = (
    ee.Image(export_image)
    .addBands(LOSS19)
    .stratifiedSample(
        numPoints=100000, classBand=LABEL, region=GEOMETRY, scale=30, tileScale=8
    )
)

randomized = sample.randomColumn()
training = randomized.filter(ee.Filter.lt("random", 0.7))
testing = randomized.filter(ee.Filter.gte("random", 0.7))

train_task = ee.batch.Export.table.toCloudStorage(
    collection=training,
    description=TRAIN_FILE_PREFIX,
    bucket=OUTPUT_BUCKET,
    fileFormat="TFRecord",
)

test_task = ee.batch.Export.table.toCloudStorage(
    collection=testing,
    description=TEST_FILE_PREFIX,
    bucket=OUTPUT_BUCKET,
    fileFormat="TFRecord",
)

# + id="l3Fq4HEtJfdM"
train_task.start()
test_task.start()

# + colab={"base_uri": "https://localhost:8080/"} id="oqtUY25tQJZ3" outputId="47a4aa39-9f8e-4628-8b7d-75f9209862ae"
train_task.status()
test_task.status()
