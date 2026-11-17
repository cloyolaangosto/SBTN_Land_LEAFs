"""Export the 10 km Google Earth Engine raster stack for the analysis grid.

The export constructs the analysis-ready raster stack used as the spatial base
for the land-degradation and yield-gap data table

"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("LDYG_DATA_ROOT", REPO_ROOT / "data"))
OUT_DIR = DATA_ROOT / "raw" / "gee_exports"
FINAL_TILE_ASSET_COLLECTION = (
    "projects/ee-hadicu06/assets/Postdoc_Bonn/analysisReadyVariablesStack/"
    "landDegradationYieldGapStudy_tiles10km"
)
FINAL_DRIVE_DESCRIPTION = (
    "landDegradationYieldGapStudy_merged10km"
)


def run_export() -> None:
    import ee

    ee.Initialize()

    # Define the palette
    viridis7Pal = '440154,3B528B,21908C,5DC863,FADE61,FDE725'

    # Define the global bounding box
    globalBbox = ee.Geometry.Polygon([[-180, 88], [0, 88], [180, 88], [180, -88], [0, -88], [-180, -88]], None, False)

    # Define the GADM and tiles FeatureCollections
    GADM = ee.FeatureCollection("users/hadicu06/Postdoc_Bonn/adminBoundaries/gadm_410-levels_simpl500m")
    # tiles = ee.FeatureCollection("users/hadicu06/Postdoc_Bonn/exportTiles/2024-01-02_globalTiles_ZanderWay_500Km_e8347ab1448158ca183c670a9998d93e")
    tiles = ee.FeatureCollection("users/hadicu06/Postdoc_Bonn/exportTiles/2024-01-02_globalTiles_ZanderWay_1000Km_639ad0432dba7611f2fec8c77590918f")

    # Define the target grid
    targetGrid = ee.Image("users/hadicu06/Postdoc_Bonn/cropYield/YieldGapTrendSummarizedGlobal/Fig1Data_AllCrops2010YieldGapPercent").rename('AllCrops2010YieldGapPercent')


    # Define the parameters
    params = {
      'debug': False,
      'isric_soil_select_depths_strings': ["_0-5cm_mean", "_5-15cm_mean", "_15-30cm_mean"],
      'studyPeriodBeginYear': 1980,   ## 1975
      'studyPeriodEndYear': 2010,
      'typicalClimateNYearsWindow': 2,
      'targetGrid': targetGrid,
      'targetProj': targetGrid.projection(),
      'targetCrs': targetGrid.projection().getInfo()['crs'],
      'targetScale': targetGrid.projection().nominalScale().getInfo(),
      'targetTransform': targetGrid.projection().getInfo()['transform'],    
      'tiles': tiles,
      'unweightedSum': True
    }

    params['reducerShortname'] = {
      'Max': ee.Reducer.max(),
      'Mean': ee.Reducer.mean(),
      'Sum': ee.Reducer.sum()
    }

    params['annualReducer'] = {
      'precipitation': 'Sum',
      'temperature': 'Mean',
      'solarRadiation': 'Mean'
    }



    def addSystemTimeProps(image):
        year = image.get('time_year_int')
        startDate = ee.Date.fromYMD(year, 1, 1)
        endDate = startDate.advance(1, 'year')
  
        startDateMillis = startDate.millis()
        endDateMillis = endDate.millis()
  
        return image.set({
            'system:time_start': startDateMillis,
            'system:time_end': endDateMillis,
        })



    def resampleReprojectOneStep(img, outProj):
        outMean = img.reduceResolution(
            reducer=ee.Reducer.mean(),   
            maxPixels=1024    
        ).reproject(
            crs=outProj
        )

        if params['unweightedSum']:
            sum_reducer = ee.Reducer.sum().unweighted()
        else:
            sum_reducer = ee.Reducer.sum()
  
        outSum = img.reduceResolution(
            reducer=sum_reducer,     
            maxPixels=1024    
        ).reproject(
            crs=outProj
        )
  
        return ee.Image.cat([
            outMean.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__mean')),    
            outSum.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__sum')),
        ])

    def resampleReprojectBinaryOneStep(img, outProj):
        area = img.reduceResolution(
            reducer=ee.Reducer.mean(),    
            maxPixels=1024   
        ).multiply(ee.Image.pixelArea()).reproject(    # area in sq-m
            crs=outProj
        )
    
        area_fraction = area.divide(ee.Image.pixelArea())

        stacked = ee.Image.cat(
            [
                area.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__area_sqm')), 
                area_fraction.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__area_fraction_fromWholePixelArea'))  
            ]
        )
  
        return stacked

    def resampleReprojectTwoSteps(img, outProj, intermScale):
        intermProj = outProj.atScale(intermScale)

        if params['unweightedSum']:
            sum_reducer = ee.Reducer.sum().unweighted()
        else:
            sum_reducer = ee.Reducer.sum()

        # Step1: source resolution (e.g., 30m) to intermediate resolution
        imgIntermProjMean = img.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=1024
        ).reproject(
            crs=intermProj
        )

        imgIntermProjSum = img.reduceResolution(
            reducer=sum_reducer,
            maxPixels=1024
        ).reproject(
            crs=intermProj
        )

        # Step2: intermediate resolution to the final resolution i.e. ~10 km
        imgOutProjMean = imgIntermProjMean.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=1024
        ).reproject(
            crs=outProj
        )

        imgOutProjSum = imgIntermProjSum.reduceResolution(
            reducer=sum_reducer,
            maxPixels=1024
        ).reproject(
            crs=outProj
        )

        return ee.Image.cat([
            imgOutProjMean.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__mean')),    
            imgOutProjSum.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__sum')),
        ])

    def resampleReprojectBinaryTwoSteps(img, outProj, intermScale):
        intermProj = outProj.atScale(intermScale)


        if params['unweightedSum']:
            sum_reducer = ee.Reducer.sum().unweighted()
        else:
            sum_reducer = ee.Reducer.sum()

        # Step1: source resolution (e.g., 30m) to intermediate resolution
        areaIntermProj = img.reduceResolution(
            reducer=ee.Reducer.mean(),    
            maxPixels=1024   
        ).multiply(ee.Image.pixelArea()).reproject(    # area in sq-m
            crs=intermProj
        )

        # Step2: intermediate resolution to the final resolution i.e. ~10 km
        areaOutProj = areaIntermProj.reduceResolution(
            reducer=sum_reducer,
            maxPixels=1024
        ).reproject(
            crs=outProj
        )

        areaOutProj_fraction = areaOutProj.divide(ee.Image.pixelArea())

        stacked = ee.Image.cat(
            [
                areaOutProj.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__area_sqm')), 
                areaOutProj_fraction.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__area_fraction_fromWholePixelArea'))  
            ]
        )

        return stacked

    def getPeriodAverage(dataImageColl, targetYear, nyearsWindow, annualReducer=None):
        targetDate = ee.Date.fromYMD(targetYear, 1, 1)

        startDate = targetDate.advance(ee.Number(nyearsWindow).multiply(-1), 'year')

        endDate = targetDate.advance(ee.Number(nyearsWindow).add(1), 'year').advance(-1, 'day')

        dateFiltered = dataImageColl.filter(ee.Filter.date(startDate, endDate))

        if annualReducer is not None:
            startYear = ee.Number(targetYear).subtract(2)
            endYear =  ee.Number(targetYear).add(2)
            years = ee.List.sequence(startYear, endYear)

            def annual_reduce(year):
                _startDate = ee.Date.fromYMD(year, 1, 1)
                _endDate = _startDate.advance(1, 'year')

                yearFiltered = dateFiltered.filter(ee.Filter.date(_startDate, _endDate))

                reduced = ee.ImageCollection(yearFiltered).reduce(
                    params['reducerShortname'][annualReducer]
                )

                return reduced

            annualReduced = years.map(annual_reduce)

            mean = ee.ImageCollection(annualReduced).reduce(ee.Reducer.mean())
        else:
            mean = dateFiltered.reduce(ee.Reducer.mean())

        meanSrcGrid = mean.setDefaultProjection(
            crs=dataImageColl.first().projection(),
            crsTransform=dataImageColl.first().projection().getInfo()['transform']
        )

        return meanSrcGrid

    def averageSoilPropsAcrossDepths(image):
        depths = params['isric_soil_select_depths_strings']

        # Get the first band name, cut the suffix _, to get e.g. "cec"
        firstBandName = ee.List(image.bandNames()).get(0)
        parts = firstBandName.getInfo().split('_')
        varName = parts[0]

        # Construct band names in the given image based on parameter `bands`
        bands = ee.List(depths).map(lambda band: ee.String(varName).cat(ee.String(band)))

        # Select those bands in the image, take per-pixel average across the bands
        averaged = image.select(bands).reduce(ee.Reducer.mean())

        averaged = averaged.rename(varName)

        # Set default projection
        averagedSrcGrid = averaged.setDefaultProjection(
            crs=image.projection(),
            crsTransform=image.projection().getInfo()['transform']
        )

        return averagedSrcGrid


    def resampleReprojectOneStepMode(img, outProj):

        imgOutProjMode = img.reduceResolution(
            reducer=ee.Reducer.mode(),
            maxPixels=1024
        ).reproject(
            crs=outProj
        )

        return ee.Image.cat([
            imgOutProjMode.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__mode')),    
        ])

    def resampleReprojectTwoStepsMode(img, outProj, intermScale):
        intermProj = outProj.atScale(intermScale)

        # Step1: source resolution (e.g., 30m) to intermediate resolution
        imgIntermProjMode = img.reduceResolution(
            reducer=ee.Reducer.mode(),
            maxPixels=1024
        ).reproject(
            crs=intermProj
        )


        # Step2: intermediate resolution to the final resolution i.e. ~10 km
        imgOutProjMode = imgIntermProjMode.reduceResolution(
            reducer=ee.Reducer.mode(),
            maxPixels=1024
        ).reproject(
            crs=outProj
        )

        return ee.Image.cat([
            imgOutProjMode.rename(ee.String(ee.List(img.bandNames()).get(0)).cat('__mode')),    
        ])


    # Define the process_export_a_tile function
    def process_export_a_tile(tile, output_image_coll, task_name, script_link, processing_parameters):
        aoi = tile.geometry()
        tile_id = tile.get('tileID').getInfo()
        cell_code = tile.get('cellCode').getInfo()

        aoiBuffered = aoi.buffer(10000)

        def clip_(image):
            return image.clip(aoiBuffered)

        def filterBounds_(imageColl):
            return imageColl.filterBounds(aoiBuffered)
    

        ##################################### Core gridded variables #################################################################


        yieldGap2010 = ee.Image("users/hadicu06/Postdoc_Bonn/cropYield/YieldGapTrendSummarizedGlobal/Fig1Data_AllCrops2010YieldGapPercent").rename('AllCrops2010YieldGapPercent')
        yieldGap2010 = clip_(yieldGap2010)

        debtErosionWater = ee.Image("users/hadicu06/Postdoc_Bonn/landDegradationDebt/wuepperBorrelli2021GCB/nativeResolution/Diff_Water_erosion").rename('Diff_Water_erosion')
        debtErosionWater = clip_(debtErosionWater)

        debtTreeCover = ee.Image("users/hadicu06/Postdoc_Bonn/landDegradationDebt/wuepperBorrelli2021GCB/nativeResolution/Diff_treeCover").rename('Diff_treeCover')
        debtTreeCover = clip_(debtTreeCover)

        soilCompactionDebt = ee.Image("users/hadicu06/Postdoc_Bonn/soilCompaction/Subsoil_compaction_susceptibility_index_SCSI").rename('Subsoil_compaction_susceptibility_index_mean')
        soilCompactionDebt = clip_(soilCompactionDebt)

        socDebt = ee.Image("users/hadicu06/Postdoc_Bonn/newLandDegradation/soc_delta_ca1983_min_ca2010_yieldGapGrid").rename('soc_delta_ca1983_min_ca2010')
        socDebt = clip_(socDebt)

        soilWaterDebt = ee.Image("users/hadicu06/Postdoc_Bonn/newLandDegradation/esa_cci_sm_1979_1983_mean_minus_2008_2012_mean").rename('sm_1979_1983_mean_minus_2008_2012_mean')
        soilWaterDebt = clip_(soilWaterDebt)


        terraClimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
        terraClimate = filterBounds_(terraClimate)
        precipitation = terraClimate.select('pr')
        temperatureMin = terraClimate.select('tmmn')
        temperatureMax = terraClimate.select('tmmx')
        solarRadiation = terraClimate.select('srad')

        precipitationBegin = getPeriodAverage(precipitation.select(['pr'], ['pr_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'], params['annualReducer']['precipitation'])
        precipitationEnd = getPeriodAverage(precipitation.select(['pr'], ['pr_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'], params['annualReducer']['precipitation'])
        temperatureMinBegin = getPeriodAverage(temperatureMin.select(['tmmn'], ['tmmn_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'], params['annualReducer']['temperature'])
        temperatureMinEnd = getPeriodAverage(temperatureMin.select(['tmmn'], ['tmmn_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'], params['annualReducer']['temperature'])
        temperatureMaxBegin = getPeriodAverage(temperatureMax.select(['tmmx'], ['tmmx_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'], params['annualReducer']['temperature'])
        temperatureMaxEnd = getPeriodAverage(temperatureMax.select(['tmmx'], ['tmmx_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'], params['annualReducer']['temperature'])
        solarRadiationBegin = getPeriodAverage(solarRadiation.select(['srad'], ['srad_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'], params['annualReducer']['solarRadiation'])
        solarRadiationEnd = getPeriodAverage(solarRadiation.select(['srad'], ['srad_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'], params['annualReducer']['solarRadiation'])


        GMTED = ee.Image('USGS/GMTED2010')
        gmted_elev = GMTED.select('be75').rename('gmted_elevation')
        gmted_elev = clip_(gmted_elev)
        gmted_elev = gmted_elev.updateMask(gmted_elev.neq(0))
        gmted_terrain = ee.Algorithms.Terrain(gmted_elev)
        gmted_slope = gmted_terrain.select('slope').rename('gmted_slope')

        geomorpho90m_tri = ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/tri")
        geomorpho90m_tri = filterBounds_(geomorpho90m_tri)
        geomorpho90m_tri_mos = geomorpho90m_tri.mosaic().rename('geomorpho90m_tri')
        geomorpho90m_tri_mos = geomorpho90m_tri_mos.setDefaultProjection(
            crs=geomorpho90m_tri.first().projection(),
            crsTransform=geomorpho90m_tri.first().projection().getInfo()['transform']
        )
        geomorpho90m_tri_mos = clip_(geomorpho90m_tri_mos)

        isric_bdod = ee.Image("projects/soilgrids-isric/bdod_mean")
        isric_cec = ee.Image("projects/soilgrids-isric/cec_mean")
        isric_cfvo = ee.Image("projects/soilgrids-isric/cfvo_mean")
        isric_clay = ee.Image("projects/soilgrids-isric/clay_mean")
        isric_sand = ee.Image("projects/soilgrids-isric/sand_mean")
        isric_silt = ee.Image("projects/soilgrids-isric/silt_mean")
        isric_nitrogen = ee.Image("projects/soilgrids-isric/nitrogen_mean")
        isric_phh20 = ee.Image("projects/soilgrids-isric/phh2o_mean")
        isric_soc = ee.Image("projects/soilgrids-isric/soc_mean")
        isric_ocd = ee.Image("projects/soilgrids-isric/ocd_mean")
        isric_ocs = ee.Image("projects/soilgrids-isric/ocs_mean")

        isric_bdod = clip_(isric_bdod)
        isric_cec = clip_(isric_cec)
        isric_cfvo = clip_(isric_cfvo)
        isric_clay = clip_(isric_clay)
        isric_sand = clip_(isric_sand)
        isric_silt = clip_(isric_silt)
        isric_nitrogen = clip_(isric_nitrogen)
        isric_phh20 = clip_(isric_phh20)
        isric_soc = clip_(isric_soc)
        isric_ocd = clip_(isric_ocd)
        isric_ocs = clip_(isric_ocs)

        isric_bdod_avg = averageSoilPropsAcrossDepths(isric_bdod)
        isric_cec_avg = averageSoilPropsAcrossDepths(isric_cec)
        isric_cfvo_avg = averageSoilPropsAcrossDepths(isric_cfvo)
        isric_clay_avg = averageSoilPropsAcrossDepths(isric_clay)
        isric_sand_avg = averageSoilPropsAcrossDepths(isric_sand)
        isric_silt_avg = averageSoilPropsAcrossDepths(isric_silt)
        isric_nitrogen_avg = averageSoilPropsAcrossDepths(isric_nitrogen)
        isric_phh20_avg = averageSoilPropsAcrossDepths(isric_phh20)
        isric_soc_avg = averageSoilPropsAcrossDepths(isric_soc)
        isric_ocd_avg = averageSoilPropsAcrossDepths(isric_ocd)
        isric_ocs_avg = isric_ocs.rename('ocs')
   
        nferCol = ee.ImageCollection("users/hadicu06/Postdoc_Bonn/fertilizer/HaNi_nfer_crop_nh4")
        nferCol = filterBounds_(nferCol)
        nferCol = nferCol.map(addSystemTimeProps)
        nferBegin = getPeriodAverage(nferCol.select(['b1'], ['nfer_crop_nh4_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'])
        nferEnd = getPeriodAverage(nferCol.select(['b1'], ['nfer_crop_nh4_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'])
        nferBegin = clip_(nferBegin)
        nferEnd = clip_(nferEnd)

        no3Col = ee.ImageCollection("users/hadicu06/Postdoc_Bonn/fertilizer/HaNi_nfer_crop_no3_ca1975_ca2010")
        no3Col = filterBounds_(no3Col)
        no3Col = no3Col.map(addSystemTimeProps)
        no3Begin = getPeriodAverage(no3Col.select(['b1'], ['nfer_crop_no3_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'])
        no3End = getPeriodAverage(no3Col.select(['b1'], ['nfer_crop_no3_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'])
        no3Begin = clip_(no3Begin)
        no3End = clip_(no3End)

        nmanureCol = ee.ImageCollection("users/hadicu06/Postdoc_Bonn/fertilizer/HaNi_nmanure_app_crop_ca1975_ca2010")
        nmanureCol = filterBounds_(nmanureCol)
        nmanureCol = nmanureCol.map(addSystemTimeProps)
        nmanureBegin = getPeriodAverage(nmanureCol.select(['b1'], ['nmanure_app_crop_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'])
        nmanureEnd = getPeriodAverage(nmanureCol.select(['b1'], ['nmanure_app_crop_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'])
        nmanureBegin = clip_(nmanureBegin)
        nmanureEnd = clip_(nmanureEnd)

        noyDepCol = ee.ImageCollection("users/hadicu06/Postdoc_Bonn/fertilizer/HaNi_ndep_noy_ca1975_ca2010")
        noyDepCol = filterBounds_(noyDepCol)
        noyDepCol = noyDepCol.map(addSystemTimeProps)
        noyDepBegin = getPeriodAverage(noyDepCol.select(['b1'], ['ndep_noy_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'])
        noyDepEnd = getPeriodAverage(noyDepCol.select(['b1'], ['ndep_noy_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'])
        noyDepBegin = clip_(noyDepBegin)
        noyDepEnd = clip_(noyDepEnd)

        nhxDepCol = ee.ImageCollection("users/hadicu06/Postdoc_Bonn/fertilizer/HaNi_ndep_nhx_ca1975_ca2010")
        nhxDepCol = filterBounds_(nhxDepCol)
        nhxDepCol = nhxDepCol.map(addSystemTimeProps)
        nhxDepBegin = getPeriodAverage(nhxDepCol.select(['b1'], ['ndep_nhx_begin']), params['studyPeriodBeginYear'], params['typicalClimateNYearsWindow'])
        nhxDepEnd = getPeriodAverage(nhxDepCol.select(['b1'], ['ndep_nhx_end']), params['studyPeriodEndYear'], params['typicalClimateNYearsWindow'])
        nhxDepBegin = clip_(nhxDepBegin)
        nhxDepEnd = clip_(nhxDepEnd)

        soilPhosphorus = ee.Image("users/hadicu06/Postdoc_Bonn/soilPhosphorus/OlsenP_kgha1_World_Aug2022_ver")
        soilPhosphorus = clip_(soilPhosphorus)
        soilPhosphorus = soilPhosphorus.rename('phosphorus_OlsenP_kgha1_World_Aug2022_ver')


        pesticide = ee.Image("users/hadicu06/Postdoc_Bonn/pesticide/APR_sum_2015_H")
        pesticide = clip_(pesticide)
        pesticide = pesticide.rename('pesticide_APR_sum_2015_H')


        irrigation = ee.ImageCollection("users/hadicu06/Postdoc_Bonn/irrigation/HID_v10")
        irrigation = filterBounds_(irrigation)

        G_AEI_2010 = ee.Image("users/hadicu06/Postdoc_Bonn/irrigation/G_AEI_2010").rename('G_AEI_2010')
        irrigation_2010 = G_AEI_2010
        irrigation_2010 = clip_(irrigation_2010)
    
        irrigation_1970 = irrigation.filter(ee.Filter.equals("time_year_num", 1970)).first()
        irrigation_1980 = irrigation.filter(ee.Filter.equals("time_year_num", 1980)).first()

        irrigation_1970 = clip_(irrigation_1970)
        irrigation_1980 = clip_(irrigation_1980)

        irrigationBegin = irrigation_1980.rename('irrigation_AEI_ha_begin')
        irrigationEnd = irrigation_2010.rename('irrigation_AEI_ha_end')


        hdi = ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/HDI_1990_2015_v2")
        hdi = clip_(hdi)
        hdi_ca2010 = hdi.select(['b19', 'b20', 'b21', 'b22', 'b23']).reduce(ee.Reducer.mean()).rename('hdi_ca2010')



        ##################################### Additional gridded variables #################################################################


        ## EVI
        # ee.ImageCollection('MODIS/061/MOD13A3')   ## 1-km, monthly
        # ee.ImageCollection("MODIS/061/MOD13A2")  ## 1-km, 16-day
        modisEVIData = (ee.ImageCollection('MODIS/061/MOD13A2')   # 16-day, so max EVI is closer to the peak
                    # .filterDate('2009-12-31', '2011-01-01')  # 2010
                    .filterDate('2007-12-31', '2013-01-01')  # 2008-2012
                    .select(['EVI', 'SummaryQA']))

        modisEVIData = filterBounds_(modisEVIData)
    

        def apply_qa_mask_modis_evi(image):
            evi = image.select('EVI')
            qa = image.select('SummaryQA')
            # SummaryQA = 0 "Good quality"
            qa_mask = qa.eq(0)
            mask = qa_mask
            return evi.updateMask(mask)
    
    
        modisEVIDataMasked = modisEVIData.map(apply_qa_mask_modis_evi)
        modisEVIDataMaskedMax = modisEVIDataMasked.reduce(ee.Reducer.max())
        modisEVIDataMaskedMax = modisEVIDataMaskedMax.setDefaultProjection(modisEVIData.first().projection())
        modisEVIDataMaskedMax = clip_(modisEVIDataMaskedMax)


        ## Gridded GDP
        gdp_ppp = ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/GDP_PPP_1990_2015_5arcmin_v2")                     ## here here
        gdp_per_capita = ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/GDP_per_capita_PPP_1990_2015_v2") 

        gdp_ppp_ca2010 = gdp_ppp.select(['b19', 'b20', 'b21', 'b22', 'b23']).reduce(ee.Reducer.mean()).rename('gdp_ppp_ca2010')
        gdp_per_capita_ca2010 = gdp_per_capita.select(['b19', 'b20', 'b21', 'b22', 'b23']).reduce(ee.Reducer.mean()).rename('gdp_per_capita_ca2010')

        gdp_ppp_ca2010 = clip_(gdp_ppp_ca2010)
        gdp_per_capita_ca2010 = clip_(gdp_per_capita_ca2010)


        ## Mechanization
        machinery = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/machinery_2008to2012_avg").rename('machinery_ca2010')

        machinery = clip_(machinery)


        ## Country level variables from Wuepper et al. 2024 
        bayesian_corruption = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/bayesian_corruption").rename('bayesian_corruption')
        property_rights_protection = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/property_rights_protection").rename('property_rights_protection')
        enforcement_imputed = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/enforcement_imputed").rename('enforcement_imputed')

        bayesian_corruption = clip_(bayesian_corruption)
        property_rights_protection = clip_(property_rights_protection)
        enforcement_imputed = clip_(enforcement_imputed)

   
        ## Government agricultural expenditure
        agriShareOfGovtExpend = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/FAO_Government_Expenditure/agriShareOfGovtExpend_ca2010").rename('agriShareOfGovtExpend_ca2010')

        agriOrientIdxForGovtExpend = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/FAO_Government_Expenditure/agriOrientIdxForGovtExpend_ca2010").rename('agriOrientIdxForGovtExpend_ca2010')


        agriShareOfGovtExpend = clip_(agriShareOfGovtExpend)
        agriOrientIdxForGovtExpend = clip_(agriOrientIdxForGovtExpend)


        ## WorldClim climatology (aggregate of 1970-2000) 
        wc_srad = ee.Image("users/hadicu06/Postdoc_Bonn/climate/wc2_1_30s_srad_yearMean").rename('wc_srad')

        wc_tavg = ee.Image("users/hadicu06/Postdoc_Bonn/climate/wc2_1_30s_tavg_yearMean").rename('wc_tavg')

        wc_prec = ee.Image("users/hadicu06/Postdoc_Bonn/climate/wc2_1_30s_prec_yearSum").rename('wc_prec')

        wc_meanTempWarmestQuarter = ee.Image("users/hadicu06/Postdoc_Bonn/climate/wc_meanTempWarmestQuarter").rename('wc_meanTempWarmestQuarter')

        wc_maxTempWarmestMonth = ee.Image("users/hadicu06/Postdoc_Bonn/climate/wc_maxTempWarmestMonth").rename('wc_maxTempWarmestMonth')

        wc_precWettestQuarter = ee.Image("users/hadicu06/Postdoc_Bonn/climate/wc_precWettestQuarter").rename('wc_precWettestQuarter')


        wc_srad = clip_(wc_srad)
        wc_tavg = clip_(wc_tavg)
        wc_prec = clip_(wc_prec)
        wc_meanTempWarmestQuarter = clip_(wc_meanTempWarmestQuarter)
        wc_maxTempWarmestMonth = clip_(wc_maxTempWarmestMonth)
        wc_precWettestQuarter = clip_(wc_precWettestQuarter)

    

        ## Soil type (USDA)
        soil_type = ee.Image("users/hadicu06/Postdoc_Bonn/soil/hengl_sol_grtgroup_usda_soltax_c_250m_v02")
        alfisols = soil_type.gte(1).And(soil_type.lte(49)).rename('alfisols')
        andisols = soil_type.gte(50).And(soil_type.lte(81)).rename('andisols')
        entisols = soil_type.gte(118).And(soil_type.lte(158)).rename('entisols')
        histosols = soil_type.gte(179).And(soil_type.lte(205)).rename('histosols')
        inceptisols = soil_type.gte(206).And(soil_type.lte(267)).rename('inceptisols')
        mollisols = soil_type.gte(268).And(soil_type.lte(313)).rename('mollisols')
        oxisols = soil_type.gte(314).And(soil_type.lte(341)).rename('oxisols')
        spodosols = soil_type.gte(342).And(soil_type.lte(369)).rename('spodosols')
        ultisols = soil_type.gte(370).And(soil_type.lte(402)).rename('ultisols')
        vertisols = soil_type.gte(403).And(soil_type.lte(434)).rename('vertisols')
        aridisols = soil_type.gte(82).And(soil_type.lte(117)).rename('aridisols')
        gelisols = soil_type.gte(159).And(soil_type.lte(178)).rename('gelisols')

        soil_types = (alfisols.multiply(1)
                    .add(andisols.multiply(2))
                    .add(entisols.multiply(3))
                    .add(histosols.multiply(4))
                    .add(inceptisols.multiply(5))
                    .add(mollisols.multiply(6))
                    .add(oxisols.multiply(7))
                    .add(spodosols.multiply(8))
                    .add(ultisols.multiply(9))
                    .add(vertisols.multiply(10))
                    .add(aridisols.multiply(11))
                    .add(gelisols.multiply(12))).rename('soil_type')


        ## Aridity 
        aridity_index_yearly = ee.Image("projects/sat-io/open-datasets/global_ai/global_ai_yearly").rename('aridity')

        aridity_index_yearly = clip_(aridity_index_yearly)



        ## Additional country-level variables data
        electricityRural = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/electricityRural_ca2010").rename('electricityRural_ca2010')

        phoneSub = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/phoneSub_ca2010").rename('phoneSub_ca2010')




        ##################################### Additional gridded variables #################################################################


        agriEmp = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/agriEmp_ca2010").rename('agriEmp_ca2010')   
        agriEmp = clip_(agriEmp)

        agriShr = ee.Image("users/hadicu06/Postdoc_Bonn/countryLevelVariables/agriShr_ca2010").rename('agriShr_ca2010')   
        agriShr = clip_(agriShr)

        # fieldSize = ee.Image("users/hadicu06/Postdoc_Bonn/fieldSize/dominant_field_size_categories_yieldGapGrid").rename('dominantFieldSizeCategory')
        fieldSize = ee.Image("users/hadicu06/Postdoc_Bonn/fieldSize/dominant_field_size_categories").rename('dominantFieldSizeCategory') ## need original resolution
        fieldSize = clip_(fieldSize)
        fieldSize = fieldSize.updateMask(fieldSize.neq(0))

        # soilDepth = ee.Image("users/hadicu06/Postdoc_Bonn/soil/average_soil_and_sedimentary-deposit_thickness_yieldGapGrid").rename('soilDepth')
        soilDepth = ee.Image("users/hadicu06/Postdoc_Bonn/soil/average_soil_and_sedimentary-deposit_thickness_int16").rename('soilDepth') ## need original resolution
        soilDepth = clip_(soilDepth)
        soilDepth = soilDepth.updateMask(soilDepth.neq(255)).updateMask(soilDepth.neq(-1))


        roadDens = ee.Image("users/hadicu06/Postdoc_Bonn/roads/grip4_total_dens_m_km2").rename('roadDens')  # 10-km
        roadDens = clip_(roadDens)

    

        ##################################### Resampling ##################################################################


        debtErosionWaterInOutputGrid = resampleReprojectOneStep(debtErosionWater, params['targetProj'])
        debtTreeCoverInOutputGrid = resampleReprojectOneStep(debtTreeCover, params['targetProj'])

        soilCompactionDebtInOutputGrid = resampleReprojectOneStep(soilCompactionDebt, params['targetProj'])
        socDebtInOutputGrid = resampleReprojectOneStep(socDebt, params['targetProj'])
        soilWaterDebtInOutputGrid = resampleReprojectOneStep(soilWaterDebt, params['targetProj'])



        precipitationBeginInOutputGrid = resampleReprojectOneStep(precipitationBegin, params['targetProj'])    
        precipitationEndInOutputGrid = resampleReprojectOneStep(precipitationEnd, params['targetProj'])
        temperatureMinBeginInOutputGrid = resampleReprojectOneStep(temperatureMinBegin, params['targetProj'])
        temperatureMinEndInOutputGrid = resampleReprojectOneStep(temperatureMinEnd, params['targetProj'])
        temperatureMaxBeginInOutputGrid = resampleReprojectOneStep(temperatureMaxBegin, params['targetProj'])
        temperatureMaxEndInOutputGrid = resampleReprojectOneStep(temperatureMaxEnd, params['targetProj'])
        solarRadiationBeginInOutputGrid = resampleReprojectOneStep(solarRadiationBegin, params['targetProj'])
        solarRadiationEndInOutputGrid = resampleReprojectOneStep(solarRadiationEnd, params['targetProj'])

        gmted_elevInOutputGrid = resampleReprojectTwoSteps(gmted_elev, params['targetProj'], 463.84)
        gmted_slopeInOutputGrid = resampleReprojectTwoSteps(gmted_slope, params['targetProj'], 463.84)
        geomorpho90m_tri_mosInOutputGrid = resampleReprojectTwoSteps(geomorpho90m_tri_mos, params['targetProj'], 720)


        isric_bdod_avgInOutputGrid = resampleReprojectTwoSteps(isric_bdod_avg, params['targetProj'], 500)
        isric_cec_avgInOutputGrid = resampleReprojectTwoSteps(isric_cec_avg, params['targetProj'], 500)
        isric_cfvo_avgInOutputGrid = resampleReprojectTwoSteps(isric_cfvo_avg, params['targetProj'], 500)
        isric_clay_avgInOutputGrid = resampleReprojectTwoSteps(isric_clay_avg, params['targetProj'], 500)
        isric_sand_avgInOutputGrid = resampleReprojectTwoSteps(isric_sand_avg, params['targetProj'], 500)
        isric_silt_avgInOutputGrid = resampleReprojectTwoSteps(isric_silt_avg, params['targetProj'], 500)
        isric_nitrogen_avgInOutputGrid = resampleReprojectTwoSteps(isric_nitrogen_avg, params['targetProj'], 500)
        isric_phh20_avgInOutputGrid = resampleReprojectTwoSteps(isric_phh20_avg, params['targetProj'], 500)
        isric_soc_avgInOutputGrid = resampleReprojectTwoSteps(isric_soc_avg, params['targetProj'], 500)
        isric_ocd_avgInOutputGrid = resampleReprojectTwoSteps(isric_ocd_avg, params['targetProj'], 500)
        isric_ocs_avgInOutputGrid = resampleReprojectTwoSteps(isric_ocs_avg, params['targetProj'], 500)


        nferBeginInOutputGrid = resampleReprojectOneStep(nferBegin, params['targetProj'])
        nferEndInOutputGrid = resampleReprojectOneStep(nferEnd, params['targetProj'])
        no3BeginInOutputGrid = resampleReprojectOneStep(no3Begin, params['targetProj'])
        no3EndInOutputGrid = resampleReprojectOneStep(no3End, params['targetProj'])
        nmanureBeginInOutputGrid = resampleReprojectOneStep(nmanureBegin, params['targetProj'])
        nmanureEndInOutputGrid = resampleReprojectOneStep(nmanureEnd, params['targetProj'])
        noyDepBeginInOutputGrid = resampleReprojectOneStep(noyDepBegin, params['targetProj'])
        noyDepEndInOutputGrid = resampleReprojectOneStep(noyDepEnd, params['targetProj'])
        nhxDepBeginInOutputGrid = resampleReprojectOneStep(nhxDepBegin, params['targetProj'])
        nhxDepEndInOutputGrid = resampleReprojectOneStep(nhxDepEnd, params['targetProj'])
    

        soilPhosphorusInOutputGrid = resampleReprojectOneStep(soilPhosphorus, params['targetProj'])  # 1-km

        pesticideInOutputGrid = resampleReprojectOneStep(pesticide, params['targetProj'])
        irrigationBeginInOutputGrid = resampleReprojectOneStep(irrigationBegin, params['targetProj'])
        irrigationEndInOutputGrid = resampleReprojectOneStep(irrigationEnd, params['targetProj'])
        hdiInOutputGrid = resampleReprojectOneStep(hdi_ca2010, params['targetProj'])
        

        precipitationDeltaInOutputGrid = precipitationEndInOutputGrid.subtract(precipitationBeginInOutputGrid)
        temperatureMinDeltaInOutputGrid = temperatureMinEndInOutputGrid.subtract(temperatureMinBeginInOutputGrid)
        temperatureMaxDeltaInOutputGrid = temperatureMaxEndInOutputGrid.subtract(temperatureMaxBeginInOutputGrid)
        solarRadiationDeltaInOutputGrid = solarRadiationEndInOutputGrid.subtract(solarRadiationBeginInOutputGrid)
        nferDeltaInOutputGrid = nferEndInOutputGrid.subtract(nferBeginInOutputGrid)
        irrigationDeltaInOutputGrid = irrigationEndInOutputGrid.subtract(irrigationBeginInOutputGrid)
        no3DeltaInOutputGrid = no3EndInOutputGrid.subtract(no3BeginInOutputGrid)
        nmanureDeltaInOutputGrid = nmanureEndInOutputGrid.subtract(nmanureBeginInOutputGrid)
        noyDepDeltaInOutputGrid = noyDepEndInOutputGrid.subtract(noyDepBeginInOutputGrid)
        nhxDepDeltaInOutputGrid = nhxDepEndInOutputGrid.subtract(nhxDepBeginInOutputGrid)

        precipitationDeltaInOutputGrid = precipitationDeltaInOutputGrid.select(['pr_end_sum_mean__mean', 'pr_end_sum_mean__sum']).rename(['pr_sum_mean__mean_delta', 'pr_sum_mean__sum_delta'])
        temperatureMinDeltaInOutputGrid = temperatureMinDeltaInOutputGrid.select(['tmmn_end_mean_mean__mean', 'tmmn_end_mean_mean__sum']).rename(['tmmn_mean_mean__mean_delta', 'tmmn_mean_mean__sum_delta'])
        temperatureMaxDeltaInOutputGrid = temperatureMaxDeltaInOutputGrid.select(['tmmx_end_mean_mean__mean', 'tmmx_end_mean_mean__sum']).rename(['tmmx_mean_mean__mean_delta', 'tmmx_mean_mean__sum_delta'])
        solarRadiationDeltaInOutputGrid = solarRadiationDeltaInOutputGrid.select(['srad_end_mean_mean__mean', 'srad_end_mean_mean__sum']).rename(['srad_mean_mean__mean_delta', 'srad_mean_mean__sum_delta'])



        nferDeltaInOutputGrid = nferDeltaInOutputGrid.select(['nfer_crop_nh4_end_mean__mean', 'nfer_crop_nh4_end_mean__sum']).rename(['nfer_crop_nh4_mean__mean_delta', 'nfer_crop_nh4_mean__sum_delta'])
        irrigationDeltaInOutputGrid = irrigationDeltaInOutputGrid.select(['irrigation_AEI_ha_end__mean', 'irrigation_AEI_ha_end__sum']).rename(['irrigation_AEI_ha__mean_delta', 'irrigation_AEI_ha__sum_delta'])
        no3DeltaInOutputGrid = no3DeltaInOutputGrid.select(['nfer_crop_no3_end_mean__mean', 'nfer_crop_no3_end_mean__sum']).rename(['nfer_crop_no3_mean__mean_delta', 'nfer_crop_no3_mean__sum_delta'])
        nmanureDeltaInOutputGrid = nmanureDeltaInOutputGrid.select(['nmanure_app_crop_end_mean__mean', 'nmanure_app_crop_end_mean__sum']).rename(['nmanure_app_crop_mean__mean_delta', 'nmanure_app_crop_mean__sum_delta'])
        noyDepDeltaInOutputGrid = noyDepDeltaInOutputGrid.select(['ndep_noy_end_mean__mean', 'ndep_noy_end_mean__sum']).rename(['ndep_noy_mean__mean_delta', 'ndep_noy_mean__sum_delta'])
        nhxDepDeltaInOutputGrid = nhxDepDeltaInOutputGrid.select(['ndep_nhx_end_mean__mean', 'ndep_nhx_end_mean__sum']).rename(['ndep_nhx_mean__mean_delta', 'ndep_nhx_mean__sum_delta'])
    


        ##################################### Additional gridded variables #################################################################


        modisEVIDataMaskedMaxInOutputGrid = resampleReprojectOneStep(modisEVIDataMaskedMax, params['targetProj'])  # 1-km

        gdp_ppp_ca2010InOutputGrid = resampleReprojectOneStep(gdp_ppp_ca2010, params['targetProj'])
        gdp_per_capita_ca2010InOutputGrid = resampleReprojectOneStep(gdp_per_capita_ca2010, params['targetProj'])


        wc_sradInOutputGrid = resampleReprojectTwoSteps(wc_srad, params['targetProj'], 2922.75)   # 927.6624232772797
        wc_tavgInOutputGrid = resampleReprojectTwoSteps(wc_tavg, params['targetProj'], 2922.75)   # 927.6624232772797
        wc_precInOutputGrid = resampleReprojectTwoSteps(wc_prec, params['targetProj'], 2922.75)   # 927.6624232772797
        wc_meanTempWarmestQuarterInOutputGrid = resampleReprojectTwoSteps(wc_meanTempWarmestQuarter, params['targetProj'], 2922.75)   # 927.6624232772797
        wc_maxTempWarmestMonthInOutputGrid = resampleReprojectTwoSteps(wc_maxTempWarmestMonth, params['targetProj'], 2922.75)   # 927.6624232772797
        wc_precWettestQuarterInOutputGrid = resampleReprojectTwoSteps(wc_precWettestQuarter, params['targetProj'], 2922.75)   # 927.6624232772797

        soil_typesInOutputGrid = resampleReprojectTwoStepsMode(soil_types, params['targetProj'], 1460.71)  # 231.91556871282302   ### mode !

        aridity_index_yearlyInOutputGrid = resampleReprojectTwoSteps(aridity_index_yearly, params['targetProj'], 2922.75)  


        ##################################### Additional gridded variables #################################################################

        fieldSizeInOutputGrid = resampleReprojectOneStepMode(fieldSize, params['targetProj'])  ## now original resolution
        soilDepthInOutputGrid = resampleReprojectOneStep(soilDepth, params['targetProj']) ## now original resolution


        roadDensInOutputGrid = resampleReprojectOneStep(roadDens, params['targetProj'])




        ##################################### Stack all the layers ##################################################################



        stacked = ee.Image.cat([
            yieldGap2010,
            debtErosionWaterInOutputGrid,
            debtTreeCoverInOutputGrid,
            soilCompactionDebtInOutputGrid,
            socDebtInOutputGrid,
            soilWaterDebtInOutputGrid,
            precipitationDeltaInOutputGrid,
            precipitationBeginInOutputGrid,
            precipitationEndInOutputGrid,
            temperatureMinDeltaInOutputGrid,
            temperatureMinBeginInOutputGrid,
            temperatureMinEndInOutputGrid,
            temperatureMaxDeltaInOutputGrid,
            temperatureMaxBeginInOutputGrid,
            temperatureMaxEndInOutputGrid,
            solarRadiationDeltaInOutputGrid,
            solarRadiationBeginInOutputGrid,
            solarRadiationEndInOutputGrid,
            gmted_elevInOutputGrid,
            gmted_slopeInOutputGrid,
            geomorpho90m_tri_mosInOutputGrid,
            isric_bdod_avgInOutputGrid,
            isric_cec_avgInOutputGrid,
            isric_cfvo_avgInOutputGrid,
            isric_clay_avgInOutputGrid,
            isric_sand_avgInOutputGrid,
            isric_silt_avgInOutputGrid,
            isric_nitrogen_avgInOutputGrid,
            isric_phh20_avgInOutputGrid,
            isric_soc_avgInOutputGrid,
            isric_ocd_avgInOutputGrid,
            isric_ocs_avgInOutputGrid,
            nferDeltaInOutputGrid,
            nferBeginInOutputGrid,
            nferEndInOutputGrid,
            no3DeltaInOutputGrid,
            no3BeginInOutputGrid,
            no3EndInOutputGrid,
            nmanureDeltaInOutputGrid,
            nmanureBeginInOutputGrid,
            nmanureEndInOutputGrid,
            noyDepDeltaInOutputGrid,
            noyDepBeginInOutputGrid,
            noyDepEndInOutputGrid,
            nhxDepDeltaInOutputGrid,
            nhxDepBeginInOutputGrid,
            nhxDepEndInOutputGrid,
            soilPhosphorusInOutputGrid,
            pesticideInOutputGrid,
            irrigationDeltaInOutputGrid,  
            irrigationBeginInOutputGrid,
            irrigationEndInOutputGrid,
            hdiInOutputGrid,
            ## Additional variables
            modisEVIDataMaskedMaxInOutputGrid,
            gdp_ppp_ca2010InOutputGrid,
            gdp_per_capita_ca2010InOutputGrid,
            wc_sradInOutputGrid,
            wc_tavgInOutputGrid,
            wc_precInOutputGrid,
            wc_meanTempWarmestQuarterInOutputGrid,
            wc_maxTempWarmestMonthInOutputGrid,
            wc_precWettestQuarterInOutputGrid,
            soil_typesInOutputGrid,
            aridity_index_yearlyInOutputGrid,
            ## Additional variables
            machinery,
            bayesian_corruption,
            property_rights_protection,
            enforcement_imputed,
            agriShareOfGovtExpend,
            agriOrientIdxForGovtExpend,
            electricityRural,
            phoneSub,
            # Additional variables
            agriEmp,
            agriShr,
            fieldSizeInOutputGrid,
            soilDepthInOutputGrid,
            roadDensInOutputGrid
      ])



        description = task_name + "__tile_" + str(tile_id)

        stacked = stacked.set('tileID', tile_id, 'cellCode', cell_code, 'scriptLink', script_link, 'processingParameters', processing_parameters)

        task = ee.batch.Export.image.toAsset(
            image=stacked,
            description=description,
            assetId=output_image_coll + "/tile_" + str(tile_id),
            region=aoiBuffered,
            crs=params['targetCrs'],
            crsTransform=params['targetTransform'],
            maxPixels=1e13,
        )

        task.start()


    tile_ids = params['tiles'].aggregate_array('tileID')

    tile_ids_client = tile_ids.getInfo()


    scriptLink = "code/01_data/01_export_gee_stack.py"

    for tile_id in tile_ids_client:
        tile = params['tiles'].filter(ee.Filter.eq("tileID", tile_id)).first()
        process_export_a_tile(
            tile,
            "projects/ee-hadicu06/assets/Postdoc_Bonn/analysisReadyVariablesStack/landDegradationYieldGapStudy_tiles10km",
            "landDegradationYieldGapStudy_10km",
            scriptLink,
            params,
        )


    exported_stacked_coll = ee.ImageCollection(
        "projects/ee-hadicu06/assets/Postdoc_Bonn/analysisReadyVariablesStack/landDegradationYieldGapStudy_tiles10km"
    )

    exported_stacked_mosaick = exported_stacked_coll.mosaic().toDouble()

    task = ee.batch.Export.image.toDrive(
        image=exported_stacked_mosaick,
        description="landDegradationYieldGapStudy_merged10km",
        folder="POSTDOC_BONN_GEE_v2",
        region=globalBbox,
        crs=params['targetCrs'],
        crsTransform=params['targetTransform'],
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
        skipEmptyTiles=True,
    )

    task.start()


def main() -> None:
    if os.environ.get("LDYG_ALLOW_GEE_EXPORT", "false").lower() != "true":
        print("GEE export is disabled by default. Set LDYG_ALLOW_GEE_EXPORT=true to start Earth Engine export tasks.")
        return

    run_export()


if __name__ == "__main__":
    main()
