# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.9.7 ('base')
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/prosperwashaya/PS_POINTS/blob/master/TensorFlow_latest.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + id="7-VYqqaLHSlY"
# from google.colab import auth
# auth.authenticate_user()

# + colab={"base_uri": "https://localhost:8080/"} id="q3petbPSIKOB" outputId="8472d2da-1ca5-4617-ce3d-4c0fde8f1e5b"
import ee
# ee.Authenticate()
ee.Initialize()

# + colab={"base_uri": "https://localhost:8080/"} id="E3C5rzRhIMxh" outputId="c167ebfd-e0d5-4bf8-926b-972f4fdabcfc"
import tensorflow as tf
print(tf.__version__)

# + colab={"base_uri": "https://localhost:8080/"} id="YkMXDrhuINzk" outputId="fb81dd01-8013-446c-c387-7395890ab587"
import folium
print(folium.__version__)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="thkxrFJzixtH" outputId="d9a92dfc-b9a1-4d48-f7e7-a2784d2ef010"
# %%capture
pip install geemap
pip install tornado

# + id="hqtqQOwfT1sd"
# // Load Sentinel-1 C-band SAR Ground Range collection (log scale, VV, descending)
FC_map2 = ee.Image('projects/ee-pwashaya9/assets/corine_area')
study = ee.FeatureCollection('users/pwashaya9/czech_shp')
# Load Sentinel-1 C-band SAR Ground Range collection (log scale, VV, descending)

FC_map = FC_map2.reduceToVectors(
  geometry = study,
  crs = FC_map2.projection(),
  scale = 30,
  geometryType = 'polygon',
  eightConnected = False,
  labelProperty = 'zone',
  maxPixels = 1e10
)

collectionVV = ee.ImageCollection('COPERNICUS/S1_GRD') \
.filter(ee.Filter.eq('instrumentMode', 'IW')) \
.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
.filterMetadata('resolution_meters', 'equals' , 10) \
.filterBounds(study) \
.select('VV')
#print(collectionVV, 'Collection VV')

# Load Sentinel-1 C-band SAR Ground Range collection (log scale, VH, descending)
collectionVH = ee.ImageCollection('COPERNICUS/S1_GRD') \
.filter(ee.Filter.eq('instrumentMode', 'IW')) \
.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
.filterMetadata('resolution_meters', 'equals' , 10) \
.filterBounds(study) \
.select('VH')
#print(collectionVH, 'Collection VH')
# Map.addLayer(study)
#Map.addLayer(FC_map)
Forest_class_mapCZ = ee.ImageCollection(FC_map)

#Filter by date
#first2014VV = collectionVV.filterDate('2014-01-01', '2014-12-31').mosaic()
#second2015VV = collectionVV.filterDate('2015-01-01', '2015-12-31').mosaic()
#third2016VV = collectionVV.filterDate('2016-01-01', '2016-12-31').mosaic()
first2014VH = collectionVH.filterDate('2014-10-01', '2014-10-30').mosaic()
second2015VH = collectionVH.filterDate('2015-10-20', '2015-12-30').mosaic()
third2016VH = collectionVH.filterDate('2016-03-01', '2016-05-30').mosaic()

#fourth2017VV = collectionVV.filterDate('2017-01-01', '2017-12-31').mosaic()
#fifth2018VV = collectionVV.filterDate('2018-01-01', '2018-12-31').mosaic()
#sixth2019VV = collectionVV.filterDate('2019-01-01', '2019-12-31').mosaic()
fourth2017VH = collectionVH.filterDate('2017-06-01', '2017-06-30').mosaic()
fifth2018VH = collectionVH.filterDate('2018-03-20', '2018-05-30').mosaic()
sixth2019VH = collectionVH.filterDate('2019-01-01', '2019-01-30').mosaic()
seventh2020VH = collectionVH.filterDate('2020-10-01', '2020-10-30').mosaic()

#composite = second2015VH.clip(newroi)
#Map.addLayer(composite, {min:-27,max:0}, 'composite2015 VH', 0)
# Display map
# Map.centerObject(study, 7)

# #Map.addLayer(first2014VV, {min:-15,max:0}, '2014 VV', 0)
# #Map.addLayer(second2015VV, {min:-15,max:0}, '2015 VV', 0)
# #Map.addLayer(third2016VV, {min:-15,max:0}, '2016 VV', 0)
# Map.addLayer(first2014VH, {'min':-27, 'max':0}, '2014 VH', 0)
# Map.addLayer(second2015VH, {'min':-27, 'max':0}, '2015 VH', 0)
# Map.addLayer(third2016VH, {'min':-27, 'max':0}, '2016 VH', 0)

# #Map.addLayer(fourth2017VV, {min:-15,max:0}, '2017 VV', 0)
# #Map.addLayer(fifth2018VV, {min:-15,max:0}, '2018 VV', 0)
# #Map.addLayer(sixth2019VV, {min:-15,max:0}, '2019 VV', 0)
# #Map.addLayer(fourth2017VH, {min:-27,max:0}, '2017 VH', 0)
# Map.addLayer(fifth2018VH, {'min':-27, 'max':0}, '2018 VH', 0)
# Map.addLayer(sixth2019VH, {'min':-27, 'max':0}, '2019 VH', 0)
# Map.addLayer(seventh2020VH, {'min':-27, 'max':0}, '2020 VH', 0)
#Map.addLayer(first2014VH.addBands(fifth2018VH).addBands(seventh2020VH),{min:-25, max:-8}, '2014/2018,2020 composite',0)

#Apply filter to reduce speckle
SMOOTHING_RADIUS = 50
#first2014VV_filtered = first2014VV.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
first2014VH_filtered = first2014VH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
#second2015VV_filtered = second2015VV.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
second2015VH_filtered = second2015VH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
#third2016VV_filtered = third2016VV.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
third2016VH_filtered = third2016VH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')

#fourth2017VV_filtered = fourth2017VV.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
fourth2017VH_filtered = fourth2017VH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
#fifth2018VV_filtered = fifth2018VV.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
fifth2018VH_filtered = fifth2018VH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
#sixth2019VV_filtered = sixth2019VV.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
sixth2019VH_filtered = sixth2019VH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')
seventh2020VH_filtered = seventh2020VH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters')

#clip the SAR layers with FC_map

clip_first2014VH_filtered = first2014VH_filtered.clip(FC_map)
featureSimple14 = ee.Feature(clip_first2014VH_filtered)

#  return feature.simplify({maxError: 100})
#
#simplifiedCol =featureSimple.map(func_fvz)

featureSimple1 = featureSimple14.simplify(
     maxError = 3000000
    )
#Map.addLayer(featureSimple1)
clip_sixth2019VH_filtered = sixth2019VH_filtered.clip(FC_map)
featureSimple19 = ee.Feature(clip_sixth2019VH_filtered)
featureSimple2 = featureSimple19.simplify(
    maxError = 3000000)

clip_fifth2018VH_filtered = fifth2018VH_filtered.clip(FC_map)
featureSimple19 = ee.Feature(clip_fifth2018VH_filtered)
featureSimple2 = featureSimple19.simplify(
    maxError=  3000000)

clip_second2015VH_filtered =second2015VH_filtered.clip(FC_map)
featureSimple19 = ee.Feature(clip_second2015VH_filtered)
featureSimple2 = featureSimple19.simplify(
    maxError=  3000000)

clip_seventh2020VH_filtered = seventh2020VH_filtered.clip(FC_map)
featureSimple19 = ee.Feature(clip_seventh2020VH_filtered)
featureSimple2 = featureSimple19.simplify(
    maxError = 3000000)

#Add the filtered image to the layers

# #Map.addLayer(first2014VH_filtered, {min:-27,max:0}, '2014 VH filtered', 0)
# Map.addLayer(second2015VH, {'min':-27, 'max':0}, '2015 VH filtered', 0)
# Map.addLayer(third2016VH_filtered, {'min':-27, 'max':0}, '2016 VH filtered', 0)

# #Map.addLayer(fourth2017VV, {min:-15,max:0}, '2017 VV', 0)
# #Map.addLayer(fifth2018VV, {min:-15,max:0}, '2018 VV', 0)
# #Map.addLayer(sixth2019VV, {min:-15,max:0}, '2019 VV', 0)
# #Map.addLayer(fourth2017VH, {min:-27,max:0}, '2017 VH', 0)
# Map.addLayer(fifth2018VH_filtered, {'min':-27, 'max':0}, '2018 VH filtered', 0)
# Map.addLayer(sixth2019VH_filtered, {'min':-27, 'max':0}, '2019 VH filtered', 0)
# Map.addLayer(seventh2020VH_filtered, {'min':-27, 'max':0}, '2020 VH filtered', 0)
# #Map.addLayer(first2014VH_filtered.addBands(fifth2018VH_filtered).addBands(seventh2020VH_filtered),{min:-25, max:-8}, '2014/2018,2020 filtered RGB',0)

# Calculate the ratio between before and after

ratio1518VH=  clip_second2015VH_filtered.subtract( clip_fifth2018VH_filtered)
#ratio1415VV= first2014VV_filtered.subtract(second2015VV_filtered)
#ratio1516VH= second2015VH_filtered.subtract(third2016VH_filtered)
#ratio1516VV= second2015VV_filtered.subtract(third2016VV_filtered)

#ratio1617VH= third2016VH_filtered.subtract(fourth2017VH_filtered)
#ratio1617VV= third2016VV_filtered.subtract(fourth2017VV_filtered)

#ratio1718VH= fourth2017VH_filtered.subtract(fifth2018VH_filtered)
#ratio1718VV= fourth2017VV_filtered.subtract(fifth2018VV_filtered)
#ratio1820VH= fifth2018VH_filtered.subtract( seventh2020VH_filtered)
ratio1820VH= clip_fifth2018VH_filtered.subtract(clip_seventh2020VH_filtered)

# #Calculate histograms for each image
# print(ui.Chart.image.histogram({'image':ratio1518VH, 'region':test_area, 'scale':500}))
# #print(ui.Chart.image.histogram({image:ratio1516VH, region:newroi, scale:300}))
# #print(ui.Chart.image.histogram({image:ratio1617VH, region:newroi, scale:300}))
# #print(ui.Chart.image.histogram({image:ratio1718VH, region:newroi, scale:300}))

# print(ui.Chart.image.histogram({'image':ratio1820VH, 'region':test_area, 'scale':500}))

# Combine the mean and standard deviation reducers.
reducers = ee.Reducer.mean().combine(
  reducer2 = ee.Reducer.stdDev(),
  sharedInputs = True
)

#Calculate the mean and standard deviation for each ratio image
stats1518 = ratio1518VH.reduceRegion(
  reducer = reducers,
  geometry = study,
  scale = 30,
  maxPixels = 60e7,
)

stats1820 = ratio1820VH.reduceRegion(
  reducer = reducers,
  geometry = study,
  scale = 30,
  maxPixels = 60e7,
  #tileScale: 16
)

#Print the mean and stdv for each ratio image
# print('stats:', stats1518, stats1820)
#, stats1516,stats1617,stats1718,stats1819
#Apply Thresholds based on stdvx6.14
#RATIO_UPPER_THRESHOLD1418 = 7.17
RATIO_UPPER_THRESHOLD1518 = 3.4
#RATIO_UPPER_THRESHOLD1516 = 6.14
#RATIO_UPPER_THRESHOLD1617 = 6.14
#RATIO_UPPER_THRESHOLD1718 = 35.78
RATIO_UPPER_THRESHOLD1820 = 2.96

ratio1518VH_thresholded = ratio1518VH.gt(RATIO_UPPER_THRESHOLD1518)
#ratio1516VH_thresholded = ratio1516VH.gt(RATIO_UPPER_THRESHOLD1516)

#ratio1617VH_thresholded = ratio1617VH.gt(RATIO_UPPER_THRESHOLD1617)
#ratio1718VH_thresholded = ratio1718VH.gt(RATIO_UPPER_THRESHOLD1718)

ratio1820VH_thresholded = ratio1820VH.gt(RATIO_UPPER_THRESHOLD1820)

#Display Masks
# Map.addLayer(ratio1518VH_thresholded.updateMask(ratio1518VH_thresholded),{'palette':"#f54009"},'Vegetation Loss 15/18',1)

# #Map.addLayer(ratio1516VH_thresholded.updateMask(ratio1516VH_thresholded),{palette:"cdb33b"},'Vegetation Loss 15/16',1)
# #Map.addLayer(ratio1617VH_thresholded.updateMask(ratio1617VH_thresholded),{palette:"cc0013"},'Vegetation Loss 16/17',1)
# #Map.addLayer(ratio1718VH_thresholded.updateMask(ratio1718VH_thresholded),{palette:"FF0000"},'Vegetation Loss 17/18',1)
# Map.addLayer(ratio1820VH_thresholded.updateMask(ratio1820VH_thresholded),{'palette':"140b13"},'Vegetation Loss 18/20',1)

#Compare differences in vegetation loss between 16/18 and 18/20
area_loss1518 = ratio1518VH_thresholded.reduceRegion(
  reducer= ee.Reducer.sum(),
  geometry= study,
  scale= 20,
  maxPixels= 60e7,
)

#Print the mean and stdv for each ratio image
# print('stats:', area_loss1518)

# Export.image.toDrive({
#    'image': ratio1518VH_thresholded,
#    'description': 'area_loss1518',
#    'scale': 10,
#    'region': newroi,
#    'fileFormat': 'GeoTIFF',
#    'maxPixels': 40e8
# })

# Export.table.toDrive({
#   'collection': ee.FeatureCollection([
#     ee.Feature(None,area_loss1518 )
#   ]),
#   'description': 'stats1518',
#   'fileFormat': 'CSV'
# })
# Map





# + id="YF05sMPx4Jse"
dataset = ee.Image('CGIAR/SRTM90_V4');
elevation = dataset.select('elevation').clip(FC_map)
slope = ee.Terrain.slope(elevation).clip(FC_map)
aspect = ee.Terrain.aspect(elevation).clip(FC_map)

# + id="ubStHOFAjJwY"
import geemap
import os

# + id="yulGmPI4IpTw"
# REPLACE WITH YOUR CLOUD PROJECT!
PROJECT = 'quick-keel-352020'

# Output bucket for trained models.  You must be able to write into this bucket.
OUTPUT_BUCKET = 'prosper-bucket'

# Cloud Storage bucket with training and testing datasets.
DATA_BUCKET = 'prosper-bucket'

# This is a good region for hosting AI models.
REGION = 'us-central1'

# Training and testing dataset file names in the Cloud Storage bucket.
TRAIN_FILE_PREFIX = 'logistic_demo_training'
TEST_FILE_PREFIX = 'logistic_demo_testing'
file_extension = '.tfrecord.gz'
TRAIN_FILE_PATH = 'gs://' + DATA_BUCKET + '/' + TRAIN_FILE_PREFIX + file_extension
TEST_FILE_PATH = 'gs://' + DATA_BUCKET + '/' + TEST_FILE_PREFIX + file_extension

# The labels, consecutive integer indices starting from zero, are stored in
# this property, set on each point.
LABEL = 'loss19'
# Number of label values, i.e. number of classes in the classification.
N_CLASSES = 2

# Study area.  CZECH.
# GEOMETRY = geemap.shp_to_ee( r'C:\Users\Prosper\Desktop\EE\GEEMAP\SPH_STAT.shp', encoding="ISO-8859-1")
GEOMETRY = ee.FeatureCollection('users/pwashaya9/czech_shp')


def compositeFunctionSR(image):
  # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
                qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image \
          .select(['B[1-7]']).multiply(0.0001) \
          .addBands(image.select(['B10', 'B11']).multiply(0.1)) \
          .updateMask(mask)
# Cloud masking function.
# def compositeFunctionSR(image):
#   cloudShadowBitMask = ee.Number(2).pow(3).int()
#   cloudsBitMask = ee.Number(2).pow(5).int()
#   qa = image.select('pixel_qa')
#   mask1 = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
#     qa.bitwiseAnd(cloudsBitMask).eq(0))
#   mask2 = image.mask().reduce('min')
#   mask3 = image.select(OPTICAL_BANDS).gt(0).And(
#           image.select(OPTICAL_BANDS).lt(10000)).reduce('min')
#   mask = mask1.And(mask2).And(mask3)
#   return image.select(OPTICAL_BANDS).divide(10000).addBands(
#           image.select(THERMAL_BANDS).divide(10).clamp(273.15, 373.15)
#             .subtract(273.15).divide(100)).updateMask(mask)


corine = ee.Image('COPERNICUS/CORINE/V18_5_1/100m/2012') \
              .clip(GEOMETRY)

# print(corine)
# print(corine.propertyNames())
lc_value = corine.get('landcover_class_names')
# print(lc_value)

forest = corine.updateMask(corine.gte(22).And(corine.lte(25)))

# Map.addLayer(forest)
def landuse(image):
  # Bits 3 and 5 are cloud shadow and cloud, respectively.
    landuse_mask = forest
    # Both flags should be set to zero, indicating clear conditions.
    return image.updateMask(landuse_mask)

image_2019 = ee.ImageCollection(['LANDSAT/LC08/C01/T1_SR/LC08_188025_20190824','LANDSAT/LC08/C01/T1_SR/LC08_188026_20190824',
                            'LANDSAT/LC08/C01/T1_SR/LC08_189025_20190612','LANDSAT/LC08/C01/T1_SR/LC08_189026_20190612',
                            'LANDSAT/LC08/C01/T1_SR/LC08_190025_20190603','LANDSAT/LC08/C01/T1_SR/LC08_190026_20190603',
                            'LANDSAT/LC08/C01/T1_SR/LC08_191025_20190626', 'LANDSAT/LC08/C01/T1_SR/LC08_191026_20190626',
                            'LANDSAT/LC08/C01/T1_SR/LC08_192025_20190617','LANDSAT/LC08/C01/T1_SR/LC08_192026_20190703',
                            'LANDSAT/LC08/C01/T1_SR/LC08_193025_20190624','LANDSAT/LC08/C01/T1_SR/LC08_193026_20190624']) \
                            .map(compositeFunctionSR) \
                            .map(landuse)
image_2015 = ee.ImageCollection(['LANDSAT/LC08/C01/T1_SR/LC08_188026_20150712','LANDSAT/LC08/C01/T1_SR/LC08_189025_20150703',
                            'LANDSAT/LC08/C01/T1_SR/LC08_189026_20150719', 'LANDSAT/LC08/C01/T1_SR/LC08_190025_20150811',
                            'LANDSAT/LC08/C01/T1_SR/LC08_190026_20150726', 'LANDSAT/LC08/C01/T1_SR/LC08_191025_20150701',
                            'LANDSAT/LC08/C01/T1_SR/LC08_191026_20150717','LANDSAT/LC08/C01/T1_SR/LC08_192025_20150606',
                            'LANDSAT/LC08/C01/T1_SR/LC08_192026_20150606','LANDSAT/LC08/C01/T1_SR/LC08_193025_20150629',
                            'LANDSAT/LC08/C01/T1_SR/LC08_193026_20150731']) \
                            .map(compositeFunctionSR) \
                            .map(landuse)

# Study area.  CZECH.
# GEOMETRY2 = geemap.shp_to_ee( r'C:\Users\Prosper\Desktop\EE\GEEMAP\SPH_STAT.shp', encoding="ISO-8859-1")
# GEOMETRY = ee.Feature('users/pwashaya9/czech_shp')

OPTICAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
NDVI_BANDS = ['NDVI']
NDWI = ['NDWI']
BANDS = [NDVI_BANDS + NDWI]


mosaic_2019 = image_2019.mosaic()
# unmasked_2019 = mosaic_2019 #.unmask(0)
final_2019 = (mosaic_2019.clip(GEOMETRY)).select(OPTICAL_BANDS)

mosaic_2015 = image_2015.mosaic()
# unmasked_2015 = mosaic_2015#.unmask(0)
final_2015= (mosaic_2015.clip(GEOMETRY)).select(OPTICAL_BANDS)

NIR_2015 = final_2015.select('B5')
RED_2015 = final_2015.select('B4')
NIR_2019 = final_2019.select('B5')
RED_2019 = final_2019.select('B4')
SWIR_2015 = final_2015.select('B7')
SWIR_2019 = final_2019.select('B7')
GREEN_2015 = final_2015.select('B3')
GREEN_2019 = final_2019.select('B3')

NDVI_2019_ =(NIR_2019.subtract(RED_2019)).divide(NIR_2019.add(RED_2019)).rename('NDVI')
NDVI_2015_ =(NIR_2015.subtract(RED_2015)).divide(NIR_2015.add(RED_2015)).rename('NDVI')

NDWI_2019_ =(GREEN_2019.subtract(NIR_2019)).divide(GREEN_2019.add(NIR_2019)).rename('NDWI')
NDWI_2015_ =(GREEN_2015.subtract(NIR_2015)).divide(GREEN_2015.add(NIR_2015)).rename('NDWI')

EVI_2015 = final_2015.expression('(2.5 * ((NIR - RED)) / (NIR + 6 * RED - 7.5 *Blue + 1 ))', {
              'NIR': final_2015.select('B5'),
              'RED': final_2015.select('B4'),
              'Blue': final_2015.select('B2')}).rename('EVI');
EVI_2019 = final_2019.expression('(2.5 * ((NIR - RED)) / (NIR + 6 * RED - 7.5 *Blue + 1 ))', {
              'NIR': final_2019.select('B5'),
              'RED': final_2019.select('B4'),
              'Blue': final_2019.select('B2')}).rename('EVI');

SAR_Layer= ratio1518VH.select('VH')

# NDVI_2019_= image_2019.expression('((NIR - RED) / (NIR + RED))', {
#               'NIR': image_2019.select('B5'),
#               'RED': image_2019.select('B4'),
#               'Blue': image_2019.select('B2')}).rename('NDVI')
# NDVI_2015_ = image_2015.expression('((NIR - RED) / (NIR + RED))', {
#               'NIR': image_2015.select('B5'),
#               'RED': image_2015.select('B4'),
#               'Blue': image_2015.select('B2')}).rename('NDVI')

# NDVI_2015 = NDVI_2015_.select(['NDVI'])
# NDVI_2019 = NDVI_2019_.select(['NDVI'])
slope_Layer = slope.select('slope')
elevation_Layer = elevation.select('elevation')
aspect_Layer = aspect.select('aspect')

# Use ndvi DIFFERENCE FOR PREDICTION
stack_NDVI = (NDVI_2019_.subtract(NDVI_2015_)).rename('NDVI')
stack_NDWI = (NDWI_2019_.subtract(NDWI_2015_)).rename('NDWI')
stack_EVI = (EVI_2019.subtract(EVI_2015)).rename('EVI')
stack_SAR = SAR_Layer.rename('SAR')
stack_slope = slope.rename('slope')
stack_elevation= elevation_Layer.rename('elevation')
stack_aspect= aspect_Layer.rename('aspect')
# stack = NDVI_Layer.lt(-0.1)

stack = ((stack_NDVI.addBands(stack_NDWI)).addBands(stack_EVI).addBands(stack_SAR)).addBands(stack_slope)


# composite1 = final_2019.addBands(final_2015)
# stack = NDVI_Layer.float()

# IMAGE = ee.Image('projects/ee-pwashaya9/assets/tezba_raster_binary2')

# LOSS_image = ee.ImageCollection([IMAGE]).map(landuse)
# stack = (((IMAGE).select('b1')).rename('NDVI')).uint8()

NDVI_BANDS = ['NDVI']
# BANDS = ('NDVI')
BEFORE_BANDS = NDVI_BANDS + OPTICAL_BANDS
AFTER_BANDS = [str(s) + '_1' for s in BEFORE_BANDS]
NDVI_BANDS = ['NDVI']
NDWI = ['NDWI']
EVI = ['EVI']
SAR_BAND = ['SAR']
SLOPE_BAND = ['slope']
ELEVATION_BAND = ['elevation']
ASPECT_BAND = ['aspect']
# BANDS = NDVI_BANDS + NDWI + EVI + SAR_BAND + SLOPE_BAND
BANDS = NDWI + SAR_BAND


# Forest loss in 2016 is what we want to predict.
IMAGE = ee.Image('projects/ee-pwashaya9/assets/tezba_raster_binary2')

LOSS_image = ee.ImageCollection([IMAGE]).map(landuse)
LOSS_19 = (((IMAGE).select('b1')).rename(LABEL)).uint8()
LOSS19 = LOSS_19.gt(0.5)

# GEOMETRY = ee.Geometry.Polygon(
#         [[[48.455920, 18.583328],
#           [51.237297,  18.845534],
#           [51.265385,  11.931759],
#           [48.410289, 11.898145]]], None, False)


GEOMETRY = ee.Geometry.Polygon(
        [[[11.891839177291098,48.51063424579321],
[19.0768977710411,48.51063424579321],
[19.0768977710411,51.22876410463182],
[11.891839177291098,51.22876410463182],
[11.891839177291098,48.51063424579321]]], None, False)

# GEOMETRY = ee.Geometry.Polygon([[[17.4615494854541,50.08656806303931],
#                                 [17.49897166562988,50.08656806303931],
#                                 [17.49897166562988,50.11387574304825],
#                                 [17.4615494854541,50.11387574304825],
#                                 [17.4615494854541,50.08656806303931]]], None, False)

# These names are used to specify properties in the export of training/testing
# data and to define the mapping between names and data when reading from
# the TFRecord file into a tf.data.Dataset.
FEATURE_NAMES = list(BANDS)
FEATURE_NAMES.append(LABEL)

# List of fixed-length features, all of which are float32.
columns = [
  tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in FEATURE_NAMES
]



# Dictionary with feature names as keys, fixed-length features as values.
FEATURES_DICT = dict(zip(FEATURE_NAMES, columns))

# Where to save the trained model.
MODEL_DIR = 'gs://' + OUTPUT_BUCKET + '/logistic_model'
# Where to save the EEified model.
EEIFIED_DIR = 'gs://' + OUTPUT_BUCKET + '/logistic_eeified'

# Name of the AI Platform model to be hosted.
MODEL_NAME = 'logistic_model5'
# Version of the AI Platform model to be hosted.
VERSION_NAME = 'v0'


# Map = geemap.Map()

# Map

# export_image = 'projects/ee-pwashaya9/logistic_image'

# image_task = ee.batch.Export.image.toAsset(
#   image = stack, 
#   description = 'logistic_image', 
#   assetId = export_image, 
#   region = GEOMETRY,
#   scale = 30,
#   maxPixels = 1e10
# )

# outputBucket = 'ee-prosper' #Change for your Cloud Storage bucket

# Export the image to an Earth Engine asset.
export_image = 'projects/ee-pwashaya9/assets/logistic_demo_imageNDWI'

image_task = ee.batch.Export.image.toAsset(
  image = stack, 
  description = 'logistic_demo_imageNDWI', 
  assetId = export_image, 
  region = GEOMETRY,
  scale = 30,
  maxPixels = 1e10
)

# + colab={"base_uri": "https://localhost:8080/", "height": 621, "referenced_widgets": ["2b179782dfa64c168af9dd4160937cfe", "6dfd6c6bfca9488698cc0fbd49bc62c5", "9e34cba6e3ee4fe9b77bee128033b641", "6c2a659ad02143c8951e18243fa10106", "11c9829c21904656ac1ec42527c58c8a", "cc23c18c07334befabe716a733a096e8", "1e546314289a4e7e850b0abdb8fdf60c", "b740913b794641c7925ead83bcc2db03", "076026ace6fe4382814b54d1773d0184", "739b2847eb7b4afaa49e4018a2cc4e80", "3725c0c7b31d419e8de81c7a0008aee8", "4c813612c7b246cebe67de306febc7f6", "fd3c7b0bea03409bae2c74328906a470", "afe183ddb9ac4e27a573f0c314d248b9", "bf93a45253f447f09688e2cb9bc45ad6", "cbcc51c45a8e4021b723e4222c661a70", "f6977029b2334cbe8639a756b5f0678f", "6c7d92a8fe3f468f9dfe13256d1c5f4e", "4eee8e05e51c43b5bf98bc6f40edf9fc", "3e1b201260de48f09fa4f165bead4b60", "a7fd024c3c6241209f7fda971dc9c153", "0b99987a808d444687b226caa71f65d6", "5a45beb59a984eeea975e24966a39198", "da6e81bd19d246e999cf9166dfe1fcc0", "886646d05af14d5ca3b5c3f5dc7b9f48", "ad19c359483942ac805d22b95c223ea1", "9e458e4b28594590852015bd5d07c340", "7e03e7771cd7401cac13e7915e5ae9c5"]} id="SY6RGrJGn6E6" outputId="a94c2033-b0ee-4bd7-d43b-6d66a8a2cd02"
Map = geemap.Map()
Map.addLayer(stack,{},'loss')
Map.addLayer(LOSS19,{},'19')
# Map.addLayer(elevation,{}, 'elevation')
Map.addLayer(slope,{}, 'slope')


Map

# + id="1AKLDI3iIqus"
image_task.start()

# + colab={"base_uri": "https://localhost:8080/"} id="Xtrfb3oTIulK" outputId="1f3f9021-812f-43c7-dea2-696e83227b1d"
image_task.status()

# + id="CCQxBUeyn8xD"
# GEOMETRY2 = ee.FeatureCollection('projects/ee-pwashaya9/assets/tezba_2018')

# + id="zNw8hjOtJcAz"
sample = ee.Image(export_image).addBands(LOSS19).stratifiedSample(
  numPoints = 100000,
  classBand = LABEL,
  region = GEOMETRY,
  scale = 30,
  tileScale = 8
)

randomized = sample.randomColumn()
training = randomized.filter(ee.Filter.lt('random', 0.7))
testing = randomized.filter(ee.Filter.gte('random', 0.7))

train_task = ee.batch.Export.table.toCloudStorage(
  collection = training,
  description = TRAIN_FILE_PREFIX,
  bucket = OUTPUT_BUCKET,
  fileFormat = 'TFRecord'
)

test_task = ee.batch.Export.table.toCloudStorage(
  collection = testing,
  description = TEST_FILE_PREFIX,
  bucket = OUTPUT_BUCKET,
  fileFormat = 'TFRecord'
)

# + id="l3Fq4HEtJfdM"
train_task.start()
test_task.start()

# + colab={"base_uri": "https://localhost:8080/"} id="oqtUY25tQJZ3" outputId="47a4aa39-9f8e-4628-8b7d-75f9209862ae"
train_task.status()
test_task.status()


# + id="287qK_04JhpT"
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
  """ Convert inputs to a tuple.

  Note that the inputs must be a tuple of tensors in the right shape.

  Args:
    dict: a dictionary of tensors keyed by input name.
    label: a tensor storing the response variable.

  Returns:
    A tuple of tensors: (predictors, label).
  """
  # Values in the tensor are ordered by the list of predictors.
  predictors = [inputs.get(k) for k in BANDS]
  return (tf.expand_dims(tf.transpose(predictors), 1),
          tf.expand_dims(tf.expand_dims(label, 1), 1)) 



# + colab={"base_uri": "https://localhost:8080/"} id="qTFY279qJk4q" outputId="6089310a-cf07-4d1d-dde9-bc99002000d7"
# Load datasets from the files.
train_dataset = tf.data.TFRecordDataset(TRAIN_FILE_PATH, compression_type='GZIP')
test_dataset = tf.data.TFRecordDataset(TEST_FILE_PATH, compression_type='GZIP')

# Compute the size of the shuffle buffer.  We can get away with this
# because it's a small dataset, but watch out with larger datasets.
train_size = 0
for _ in iter(train_dataset):
  train_size+=1

batch_size = 8

# Map the functions over the datasets to parse and convert to tuples.
train_dataset = train_dataset.map(parse_tfrecord, num_parallel_calls=4)
train_dataset = train_dataset.map(to_tuple, num_parallel_calls=4)
train_dataset = train_dataset.shuffle(train_size).batch(batch_size)

test_dataset = test_dataset.map(parse_tfrecord, num_parallel_calls=4)
test_dataset = test_dataset.map(to_tuple, num_parallel_calls=4)
test_dataset = test_dataset.batch(batch_size)

# Print the first parsed record to check.
from pprint import pprint
pprint(iter(train_dataset).next())

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="q5miK7LJQ1e7" outputId="990622f1-61d3-4152-e80d-d8e34998f997"
pip install tensorflow_decision_forests

# + id="qqVXROcKQzr0"


# + colab={"base_uri": "https://localhost:8080/"} id="_X5I6Zy6Jnsj" outputId="09738b0e-80a8-4242-fe62-9af79797b9bf"
from tensorflow import keras

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Dense(5, activation='sigmoid'), #identity
#   tf.keras.layers.Dense(5, activation='sigmoid'),
#   tf.keras.layers.Dense(5, activation='sigmoid'),
#   tf.keras.layers.Dense(1)])

# model = tf.keras.models.Sequential(RandomForestModel(num_trees=30))

# Define the layers in the model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Input((1, 1, len(BANDS))),
  tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid'),
  tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid'),
  tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')
])


# Compile the model with the specified loss function.
model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit the model to the training data.
model.fit(x=train_dataset, 
          epochs=20,
          validation_data=test_dataset)

# + colab={"base_uri": "https://localhost:8080/"} id="wn2cMW39VMth" outputId="e94c72b6-dcd4-46ab-b658-554e0b230413"
model.save(MODEL_DIR, save_format='tf')

# + colab={"base_uri": "https://localhost:8080/"} id="Mt--6MCHVaJu" outputId="259b9bd7-547c-4e22-9c29-3db6dbf5cdfd"
from tensorflow.python.tools import saved_model_utils

meta_graph_def = saved_model_utils.get_meta_graph_def(MODEL_DIR, 'serve')
inputs = meta_graph_def.signature_def['serving_default'].inputs
outputs = meta_graph_def.signature_def['serving_default'].outputs

# Just get the first thing(s) from the serving signature def.  i.e. this
# model only has a single input and a single output.
input_name = None
for k,v in inputs.items():
  input_name = v.name
  break

output_name = None
for k,v in outputs.items():
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
PROJECT = 'quick-keel-352020'

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
    proj=ee.Projection('EPSG:4326').atScale(30),
    fixInputProj=True,
    # Note the names here need to match what you specified in the
    # output dictionary you passed to the EEifier.
    outputBands={'output': {
        'type': ee.PixelType.float(),
        'dimensions': 1
      }
    },
)

# Output probability.
predictions = model.predictImage(array_image).arrayGet([0])

# Back-of-the-envelope decision rule.
predicted = predictions.gt(0.5).selfMask()

# Training data for comparison.
reference = LOSS19.selfMask()

# Get map IDs for display in folium.
probability_vis = {'min': 0, 'max': 1}
probability_mapid = predictions.getMapId(probability_vis)

predicted_vis = {'palette': 'red'}
predicted_mapid = predicted.getMapId(predicted_vis)

reference_vis = {'palette': 'orange'}
reference_mapid = reference.getMapId(reference_vis)

image_vis = {'bands': ['NDVI'], 'min': 0, 'max': 0.3}
image_mapid = NDVI_2019_.getMapId(image_vis)

# Visualize the input imagery and the predictions.
map = folium.Map(location=[-9.1, -62.3], zoom_start=11)
folium.TileLayer(
  tiles=image_mapid['tile_fetcher'].url_format,
  attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
  overlay=True,
  name='image',
).add_to(map)
folium.TileLayer(
  tiles=probability_mapid['tile_fetcher'].url_format,
  attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
  overlay=True,
  name='probability',
).add_to(map)
folium.TileLayer(
  tiles=predicted_mapid['tile_fetcher'].url_format,
  attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
  overlay=True,
  name='predicted',
).add_to(map)
folium.TileLayer(
  tiles=reference_mapid['tile_fetcher'].url_format,
  attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
  overlay=True,
  name='reference',
).add_to(map)
map.add_child(folium.LayerControl())
map

# + id="fB2-SQM3hArp"
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

# + id="pTamnp-EhQ3d"
import numpy as np


# + id="SuSB8aeRiNRQ"
y_test=LOSS19.toArray()

# + colab={"base_uri": "https://localhost:8080/"} id="Ra4qnmUJjv5M" outputId="b85ba1e4-4faa-4726-97b8-3b3ccb0a3f7e"
print(y_test)

# + colab={"base_uri": "https://localhost:8080/", "height": 433} id="jJiSAp7ahQ7W" outputId="294fb4aa-91ef-4416-a31a-82b92b8991e2"
np.sqrt(mean_squared_error(test_dataset, predictions))
