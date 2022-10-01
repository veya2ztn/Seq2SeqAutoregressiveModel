year=2018
for vname in 10m_u_component_of_wind 10m_v_component_of_wind 2m_temperature surface_pressure mean_sea_level_pressure 1000h_u_component_of_wind 1000h_v_component_of_wind 1000h_geopotential 850h_temperature 850h_u_component_of_wind 850h_v_component_of_wind 850h_geopotential 850h_relative_humidity 500h_temperature 500h_u_component_of_wind 500h_v_component_of_wind 500h_geopotential 500h_relative_humidity 50h_geopotential total_column_water_vapour;
    do ~/bin/sensesync sync s3://FCM1NI7IC4S78J2EGJUC:zRQ5lOMyaXWdEjbqt24rcIQD9wZilMhn9v45wbPo@era5npy.10.140.2.254:80/$vname/$year/ /nvme/zhangtianning/datasets/ERA5/$vname/$year/;
done;
