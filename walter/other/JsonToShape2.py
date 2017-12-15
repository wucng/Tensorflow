# -*- coding:utf-8 -*-

"""
json 转 shp
"""

from osgeo import ogr
import glob
import os
import warnings
warnings.filterwarnings("ignore") # 忽略警告信息

# 打开json文件
geojson=r"E:\json"
# geojson=r"E:\hubei3719"
# geojson=input("输入json目录：")

geojsons=glob.glob(geojson+r"\*.json")


# Create the output Driver
outDriver = ogr.GetDriverByName('ESRI Shapefile')

# Create the output shp
# shp=input("输入shp存放位置：")
shp=geojson+".shp"
outDataSource = outDriver.CreateDataSource(shp)
outLayer = outDataSource.CreateLayer('test', geom_type=ogr.wkbMultiPolygon)
# Add an ID field
idField = ogr.FieldDefn("name", ogr.OFTString)
idField.SetWidth(100)
outLayer.CreateField(idField)

# outLayer=None
flag=True
m=1
for geojson in geojsons:
    print("转换中：%.2f%%" % (m/len(geojsons) * 100), end="\r")
    geojson_ds=ogr.Open(geojson,0)

    for i in range(geojson_ds.GetLayerCount()):
        layer=geojson_ds.GetLayerByIndex(i)
        feature = layer.GetNextFeature()
        # layer=geojson_ds.GetLayer()
        if flag==True:
            # outLayer = outDataSource.CopyLayer(layer, layer.GetName())
            for j in range(feature.GetFieldCount()-1):
                idField = ogr.FieldDefn(feature.GetFieldDefnRef(1+j).name, feature.GetFieldDefnRef(j+1).type)
                idField.SetWidth(100)
                outLayer.CreateField(idField)
            flag=False
        # feature=layer.GetFeature()

        while feature!=None:
            # feature.SetFiled(feature.GetFieldDefnRef(1+j).name,)
            outLayer.CreateFeature(feature)
            feature = layer.GetNextFeature()
    # featureDefn1 = layer.GetLayerDefn()

    # outLayer = outDataSource.CreateLayer('test.shp', geom_type=ogr.wkbPolygon)
    # outLayer=outDataSource.CopyLayer(layer,layer.GetName())
    # Get the output Layer's Feature Definition
    # featureDefn = outLayer.GetLayerDefn()

    # create a new feature
    # outFeature = ogr.Feature(featureDefn1)

    # Set new geometry
    # outFeature.SetGeometry(poly)

    # Add new feature to output Layer
    # outLayer.CreateFeature(outFeature)

    # dereference the feature
    # outFeature = None

    # Save and close DataSources

    geojson_ds=None
    m+=1
print("转换完成：100.00%", end="\r")
outDataSource = None
