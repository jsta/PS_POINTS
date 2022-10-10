import ee
import tensorflow as tf


def compositeFunctionSR(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select("pixel_qa")
    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    )
    return (
        image.select(["B[1-7]"])
        .multiply(0.0001)
        .addBands(image.select(["B10", "B11"]).multiply(0.1))
        .updateMask(mask)
    )


def landuse(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    landuse_mask = forest
    # Both flags should be set to zero, indicating clear conditions.
    return image.updateMask(landuse_mask)


OUTPUT_BUCKET = "rabpro-gee-uploads"

# ee.Authenticate()
ee.Initialize()

# + id="hqtqQOwfT1sd"
# // Load Sentinel-1 C-band SAR Ground Range collection (log scale, VV, descending)
FC_map = ee.FeatureCollection("projects/jsta-pspoints/assets/czech_shp")
study = ee.FeatureCollection("projects/jsta-pspoints/assets/czech_shp")

# Load Sentinel-1 C-band SAR Ground Range collection (log scale, VH, descending)
collectionVH = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
    .filterMetadata("resolution_meters", "equals", 10)
    .filterBounds(study)
    .select("VH")
)

# Filter by date
first2014VH = collectionVH.filterDate("2014-10-01", "2014-10-30").mosaic()
second2015VH = collectionVH.filterDate("2015-10-20", "2015-12-30").mosaic()
third2016VH = collectionVH.filterDate("2016-03-01", "2016-05-30").mosaic()
fourth2017VH = collectionVH.filterDate("2017-06-01", "2017-06-30").mosaic()
fifth2018VH = collectionVH.filterDate("2018-03-20", "2018-05-30").mosaic()
sixth2019VH = collectionVH.filterDate("2019-01-01", "2019-01-30").mosaic()
seventh2020VH = collectionVH.filterDate("2020-10-01", "2020-10-30").mosaic()

# Apply filter to reduce speckle
SMOOTHING_RADIUS = 50
first2014VH_filtered = first2014VH.focal_mean(SMOOTHING_RADIUS, "circle", "meters")
second2015VH_filtered = second2015VH.focal_mean(SMOOTHING_RADIUS, "circle", "meters")
third2016VH_filtered = third2016VH.focal_mean(SMOOTHING_RADIUS, "circle", "meters")
fourth2017VH_filtered = fourth2017VH.focal_mean(SMOOTHING_RADIUS, "circle", "meters")
fifth2018VH_filtered = fifth2018VH.focal_mean(SMOOTHING_RADIUS, "circle", "meters")
sixth2019VH_filtered = sixth2019VH.focal_mean(SMOOTHING_RADIUS, "circle", "meters")
seventh2020VH_filtered = seventh2020VH.focal_mean(SMOOTHING_RADIUS, "circle", "meters")

# clip the SAR layers with FC_map
clip_first2014VH_filtered = first2014VH_filtered.clip(FC_map)
featureSimple14 = ee.Feature(clip_first2014VH_filtered)

featureSimple1 = featureSimple14.simplify(maxError=3000000)
clip_sixth2019VH_filtered = sixth2019VH_filtered.clip(FC_map)
featureSimple19 = ee.Feature(clip_sixth2019VH_filtered)
featureSimple2 = featureSimple19.simplify(maxError=3000000)

clip_fifth2018VH_filtered = fifth2018VH_filtered.clip(FC_map)
featureSimple19 = ee.Feature(clip_fifth2018VH_filtered)
featureSimple2 = featureSimple19.simplify(maxError=3000000)

clip_second2015VH_filtered = second2015VH_filtered.clip(FC_map)
featureSimple19 = ee.Feature(clip_second2015VH_filtered)
featureSimple2 = featureSimple19.simplify(maxError=3000000)

clip_seventh2020VH_filtered = seventh2020VH_filtered.clip(FC_map)
featureSimple19 = ee.Feature(clip_seventh2020VH_filtered)
featureSimple2 = featureSimple19.simplify(maxError=3000000)


# Calculate the ratio between before and after
ratio1518VH = clip_second2015VH_filtered.subtract(clip_fifth2018VH_filtered)
ratio1820VH = clip_fifth2018VH_filtered.subtract(clip_seventh2020VH_filtered)

# Combine the mean and standard deviation reducers.
reducers = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)

# Calculate the mean and standard deviation for each ratio image
stats1518 = ratio1518VH.reduceRegion(
    reducer=reducers,
    geometry=study,
    scale=30,
    maxPixels=60e7,
)

stats1820 = ratio1820VH.reduceRegion(
    reducer=reducers,
    geometry=study,
    scale=30,
    maxPixels=60e7,
    # tileScale: 16
)

# Apply Thresholds based on stdvx6.14
RATIO_UPPER_THRESHOLD1518 = 3.4
RATIO_UPPER_THRESHOLD1820 = 2.96
ratio1518VH_thresholded = ratio1518VH.gt(RATIO_UPPER_THRESHOLD1518)
ratio1820VH_thresholded = ratio1820VH.gt(RATIO_UPPER_THRESHOLD1820)

# + id="YF05sMPx4Jse"
dataset = ee.Image("CGIAR/SRTM90_V4")
elevation = dataset.select("elevation").clip(FC_map)
slope = ee.Terrain.slope(elevation).clip(FC_map)
aspect = ee.Terrain.aspect(elevation).clip(FC_map)

# The labels, consecutive integer indices starting from zero, are stored in
# this property, set on each point.
LABEL = "loss19"
N_CLASSES = 2

GEOMETRY = ee.FeatureCollection("projects/jsta-pspoints/assets/czech_shp")

corine = ee.Image("COPERNICUS/CORINE/V18_5_1/100m/2012").clip(GEOMETRY)
forest = corine.updateMask(corine.gte(22).And(corine.lte(25)))

image_2019 = (
    ee.ImageCollection(
        [
            "LANDSAT/LC08/C01/T1_SR/LC08_188025_20190824",
            "LANDSAT/LC08/C01/T1_SR/LC08_188026_20190824",
            "LANDSAT/LC08/C01/T1_SR/LC08_189025_20190612",
            "LANDSAT/LC08/C01/T1_SR/LC08_189026_20190612",
            "LANDSAT/LC08/C01/T1_SR/LC08_190025_20190603",
            "LANDSAT/LC08/C01/T1_SR/LC08_190026_20190603",
            "LANDSAT/LC08/C01/T1_SR/LC08_191025_20190626",
            "LANDSAT/LC08/C01/T1_SR/LC08_191026_20190626",
            "LANDSAT/LC08/C01/T1_SR/LC08_192025_20190617",
            "LANDSAT/LC08/C01/T1_SR/LC08_192026_20190703",
            "LANDSAT/LC08/C01/T1_SR/LC08_193025_20190624",
            "LANDSAT/LC08/C01/T1_SR/LC08_193026_20190624",
        ]
    )
    .map(compositeFunctionSR)
    .map(landuse)
)
image_2015 = (
    ee.ImageCollection(
        [
            "LANDSAT/LC08/C01/T1_SR/LC08_188026_20150712",
            "LANDSAT/LC08/C01/T1_SR/LC08_189025_20150703",
            "LANDSAT/LC08/C01/T1_SR/LC08_189026_20150719",
            "LANDSAT/LC08/C01/T1_SR/LC08_190025_20150811",
            "LANDSAT/LC08/C01/T1_SR/LC08_190026_20150726",
            "LANDSAT/LC08/C01/T1_SR/LC08_191025_20150701",
            "LANDSAT/LC08/C01/T1_SR/LC08_191026_20150717",
            "LANDSAT/LC08/C01/T1_SR/LC08_192025_20150606",
            "LANDSAT/LC08/C01/T1_SR/LC08_192026_20150606",
            "LANDSAT/LC08/C01/T1_SR/LC08_193025_20150629",
            "LANDSAT/LC08/C01/T1_SR/LC08_193026_20150731",
        ]
    )
    .map(compositeFunctionSR)
    .map(landuse)
)

OPTICAL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
NDVI_BANDS = ["NDVI"]
NDWI = ["NDWI"]
BANDS = [NDVI_BANDS + NDWI]

mosaic_2019 = image_2019.mosaic()
final_2019 = (mosaic_2019.clip(GEOMETRY)).select(OPTICAL_BANDS)
mosaic_2015 = image_2015.mosaic()
final_2015 = (mosaic_2015.clip(GEOMETRY)).select(OPTICAL_BANDS)

NIR_2015 = final_2015.select("B5")
RED_2015 = final_2015.select("B4")
NIR_2019 = final_2019.select("B5")
RED_2019 = final_2019.select("B4")
SWIR_2015 = final_2015.select("B7")
SWIR_2019 = final_2019.select("B7")
GREEN_2015 = final_2015.select("B3")
GREEN_2019 = final_2019.select("B3")

NDVI_2019_ = (NIR_2019.subtract(RED_2019)).divide(NIR_2019.add(RED_2019)).rename("NDVI")
NDVI_2015_ = (NIR_2015.subtract(RED_2015)).divide(NIR_2015.add(RED_2015)).rename("NDVI")

NDWI_2019_ = (
    (GREEN_2019.subtract(NIR_2019)).divide(GREEN_2019.add(NIR_2019)).rename("NDWI")
)
NDWI_2015_ = (
    (GREEN_2015.subtract(NIR_2015)).divide(GREEN_2015.add(NIR_2015)).rename("NDWI")
)

EVI_2015 = final_2015.expression(
    "(2.5 * ((NIR - RED)) / (NIR + 6 * RED - 7.5 *Blue + 1 ))",
    {
        "NIR": final_2015.select("B5"),
        "RED": final_2015.select("B4"),
        "Blue": final_2015.select("B2"),
    },
).rename("EVI")
EVI_2019 = final_2019.expression(
    "(2.5 * ((NIR - RED)) / (NIR + 6 * RED - 7.5 *Blue + 1 ))",
    {
        "NIR": final_2019.select("B5"),
        "RED": final_2019.select("B4"),
        "Blue": final_2019.select("B2"),
    },
).rename("EVI")

SAR_Layer = ratio1518VH.select("VH")

slope_Layer = slope.select("slope")
elevation_Layer = elevation.select("elevation")
aspect_Layer = aspect.select("aspect")

# Use ndvi DIFFERENCE FOR PREDICTION
stack_NDVI = (NDVI_2019_.subtract(NDVI_2015_)).rename("NDVI")
stack_NDWI = (NDWI_2019_.subtract(NDWI_2015_)).rename("NDWI")
stack_EVI = (EVI_2019.subtract(EVI_2015)).rename("EVI")
stack_SAR = SAR_Layer.rename("SAR")
stack_slope = slope.rename("slope")
stack_elevation = elevation_Layer.rename("elevation")
stack_aspect = aspect_Layer.rename("aspect")
# stack = NDVI_Layer.lt(-0.1)

stack = (
    (stack_NDVI.addBands(stack_NDWI)).addBands(stack_EVI).addBands(stack_SAR)
).addBands(stack_slope)


NDVI_BANDS = ["NDVI"]
BEFORE_BANDS = NDVI_BANDS + OPTICAL_BANDS
AFTER_BANDS = [str(s) + "_1" for s in BEFORE_BANDS]
NDVI_BANDS = ["NDVI"]
NDWI = ["NDWI"]
EVI = ["EVI"]
SAR_BAND = ["SAR"]
SLOPE_BAND = ["slope"]
ELEVATION_BAND = ["elevation"]
ASPECT_BAND = ["aspect"]
BANDS = NDWI + SAR_BAND

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

# These names are used to specify properties in the export of training/testing
# data and to define the mapping between names and data when reading from
# the TFRecord file into a tf.data.Dataset.
FEATURE_NAMES = list(BANDS)
FEATURE_NAMES.append(LABEL)

# List of fixed-length features, all of which are float32.
columns = [tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in FEATURE_NAMES]

# Dictionary with feature names as keys, fixed-length features as values.
FEATURES_DICT = dict(zip(FEATURE_NAMES, columns))
MODEL_DIR = "gs://" + OUTPUT_BUCKET + "/logistic_model"
EEIFIED_DIR = "gs://" + OUTPUT_BUCKET + "/logistic_eeified"
MODEL_NAME = "logistic_model5"
VERSION_NAME = "v0"

# Export the image to an Earth Engine asset.
export_image = "projects/jsta-pspoints/assets/logistic_demo_imageNDWI"

image_task = ee.batch.Export.image.toAsset(
    image=stack,
    description="logistic_demo_imageNDWI",
    assetId=export_image,
    region=GEOMETRY,
    scale=30,
    maxPixels=1e10,
)

# + id="1AKLDI3iIqus"
image_task.start()

# + colab={"base_uri": "https://localhost:8080/"} id="Xtrfb3oTIulK" outputId="1f3f9021-812f-43c7-dea2-696e83227b1d"
image_task.status()
