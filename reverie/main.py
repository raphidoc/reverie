import reverie.sensor.wise as wise

if __name__ == '__main__':

    img = wise.image.PixWISE(image_dir='/D/Data/TEST/TEST_WISE/MC-50/', image_name='190818_MC-50A-WI-2x1x1_v02')
    img.cal_viewing_geo()
    img.to_netcdf()
