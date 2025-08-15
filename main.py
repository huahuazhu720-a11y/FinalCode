import NyStep1_MergeTaxiZone
import NyStep2_ScrapData_newYork
import NyStep3_clean_green_yellow_forhire_data
import NyStep4_MapCensusBlockShapToZone_V1
import NyStep5_Merg_taxidata_to_zone
import NyStep6_FindConnectionNodes
import NyStep7_MapOdDataToZone
import NyStep8_MapCensusMainDataToZone
import NyStep9_DataProcess_NewYork
import NyStep10_Get_Road_Capacity
import NyStep11_0Prepare_data_for_getting_parameters
import NyStep11_1Parameter_estimation_v4_using
import NyStep12_DrawGraph
import NyStep13_get_network_manhaton_only
import NyStep14_0correct_population_on_node
import NyStep14_5prepare_data_for_simulation
import NyStep14_6DrawGraph
import NyStep15_getIterationData
import NYstep16_manhattan_simulation_final0424
import NYstep17_manhattan_simulation_conjestion_pricing
import NYstep18_conjestion_pricing_ananlys

if __name__ == "__main__":    
    """
    Step 1
    Merge the original 200+ taxi zones into 130 zones.
    Outputs:
    Merged shapefile: shapfile/merged_taxi_zones.shp
    Merged map: merged_taxi_zones.pdf
    Zone mapping files: merged_taxi_zones.pkl and merged_taxi_zones.csv (contain the mapping between the new zone_id and the original taxizone_id).
    """
    NyStep1_MergeTaxiZone.run()
    """
    Step 2
    Code for downloading data from S0801 and storing it into the GreYelHir.db database, table SensusData_NewYork.
    """
    NyStep2_ScrapData_newYork.run()
    """
    Download the 2022 monthly data (January to December) for Green Taxi, Yellow Taxi, and For-Hire Vehicles.
    After downloading, clean the data by:
    Keeping only rows with complete data.
    Further filtering rows where pickup_datetime and dropoff_datetime are both between 6:00 AM and 10:00 AM.
    Create two datasets:
    taxi: containing Green and Yellow Taxi data.
    for_hire_vehicles: containing For-Hire Vehicles data.
    """
    NyStep3_clean_green_yellow_forhire_data.run()
    """
    step4
    Overlay the merged taxi zone shapefile with the census block shapefile to identify the relationship between each merged zone and the census blocks.
    Result: grouped_taxi_zones.pkl, showing the mapping between merged zones and census block IDs.
    """
    NyStep4_MapCensusBlockShapToZone_V1.run()
    """
    step5  Map the taxi trip data (originally between 200+ zones) to the new dataset showing trips between the 130 merged zones.
    """
    NyStep5_Merg_taxidata_to_zone.run()
    """
    step6 find adjacent nodes—only neighboring areas with a connecting path can be linked.
    Result: adjacent_districts, a dictionary where each key is a node ID and the value is a list of node IDs it can connect to.
    """
    NyStep6_FindConnectionNodes.run()
    """
    step 7
    Map the original OD data (originally between Census blocks) to the newly defined zones.
    Results: two tables—ZoneOutNodes and ZoneInNodes.
    zone_id represents a node.
    OutZone_id in ZoneOutNodes represents destinations from zone_id.
    InZone_id in ZoneInNodes represents origins going to zone_id.
    count_of_jobs indicates the number of people traveling between the OD pair.
    """
    NyStep7_MapOdDataToZone.run()
    """
    step8
    Map the data downloaded from S0801 to the new zones.
    Result: ZoneSensusData_NewYork, showing the original S0801 data under the new zone definitions.
    """
    NyStep8_MapCensusMainDataToZone.run()
    """
    step 9
    Generate the final network graph for each year.
    Note: Much of the data comes from Graph_NewYork_2010.pkl, an intermediate version. The code that originally generated this file has been replaced. Using this file speeds up execution. If the file is missing, the data can still be obtained from previous steps or directly queried from the database.
    """
    NyStep9_DataProcess_NewYork.run()
    """
    step10 
    Calculate the traffic capacity of each road for every year (2022 used as an example) and update the network graph accordingly.
    """
    NyStep10_Get_Road_Capacity.run()
    """
    step11_0
    Prepares data for parameter estimation.
    """
    NyStep11_0Prepare_data_for_getting_parameters.Prepare_data_for_getting_parameters_multiprocessing()
    """
    step11_1
    parameter estimation(whole NY network)
    """
    NyStep11_1Parameter_estimation_v4_using.run()
    """
    step12
    Generates PDFs verison of map&network picutre from the network graph. Can create undirected graphs, directed graphs, and road graphs from the saved network .pkl file.
    """
    NyStep12_DrawGraph.run()
    """
    step13
    get network graph only the traffice flow related to Manhaton 
    """
    NyStep13_get_network_manhaton_only.run()
    
    """
    step 14
    Correct the population on each node, make sure the population is only related the traffice flow to Manhaton
    """
    NyStep14_0correct_population_on_node.run()
    """
    step 14_5
    prepare data for simulation
    """
    NyStep14_5prepare_data_for_simulation.run()
    """
    step 14_6
    draw the map&network picture base on the only Manhaton traffic flow related network data
    """
    NyStep14_6DrawGraph.run()

        
    """
    step 15
    get iteration data
    """
    NyStep15_getIterationData.run()

    """
    step 16
    simulation only the traffic flow related to manhatton
    """
    NYstep16_manhattan_simulation_final0424.run()

    """
    step 17
    simulation without conjestion price on only the traffic flow related to manhatton
    """
    NYstep17_manhattan_simulation_conjestion_pricing.run()

    """
    step 18
    analysis within different conjestion price on only the traffic flow related to manhatton
    """
    NYstep18_conjestion_pricing_ananlys.ananlys()

