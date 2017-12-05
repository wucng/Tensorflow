#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
矢量 转栅格，画出水位边界
'''

import sys
from osgeo import gdal, ogr, gdalconst

class RasterizeLine(object):

    def rasterizeLine(self, src_raster, dst_raster, shape):
        '''
        :param src_raster: 输入栅格影像
        :param dst_raster: 输出栅格影像
        :param shape: shp文件
        :return:
        '''
        src = gdal.Open(src_raster, gdalconst.GA_ReadOnly)
        shp = ogr.Open(shape)
        shp_layer = shp.GetLayer()

        drv = gdal.GetDriverByName('GTiff')# 获取栅格驱动
        if not drv:
            print('No "GTiff" driver', file = sys.stderr)
            src = None
            shp = None
            shp_layer = None
            return False

        target_ds = drv.CreateCopy(dst_raster, src)# 复制输入栅格影像
        # 原始影像4个波段,前3个波段是RGB
        gdal.RasterizeLayer(target_ds, [1,2,3], shp_layer, burn_values = [255, 0, 0])# 矢量栅格化

        target_ds = None
        src = None
        shp = None
        shp_layer = None
        return True

if __name__ == "__main__":
    '''
    if len(sys.argv) != 4:
        print("{} <source> <destination> <shape>".format(sys.argv[0]))
        sys.exit(1)
    rl = RasterizeLine()
    if not rl.rasterizeLine(sys.argv[1], sys.argv[2], sys.argv[3]):
        sys.exit(1)
    '''
    rl = RasterizeLine()
    rl.rasterizeLine(r"F:\PL\2\34561.tif",r"F:\PL\2\34561_shp.tif",r"F:\PL\2\shp\3456123.shp")

