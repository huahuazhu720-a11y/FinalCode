
"""
Step 1
Merge the original 200+ taxi zones into 130 zones.
Outputs:
Merged shapefile: shapfile/merged_taxi_zones.shp
Merged map: merged_taxi_zones.pdf
Zone mapping files: merged_taxi_zones.pkl and merged_taxi_zones.csv (contain the mapping between the new zone_id and the original taxizone_id).
"""
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Load the taxi zones shapefile
def run():
    gdf =  gpd.read_file(r"shapfile\taxi_zones.shp")
    remove_objectids = [1, 103, 104]
    gdf = gdf[~gdf["OBJECTID"].isin(remove_objectids)].reset_index(drop=True)
    # Convert geometries to centroids for clustering
    gdf["centroid"] = gdf.geometry.centroid
    centroids = np.array([(p.x, p.y) for p in gdf.centroid])

    # Ensure that each taxi zone has a borough column (adjust if the column name is different)
    borough_column = "borough"  # Update with the actual column name if different
    gdf["borough"] = gdf[borough_column]

    # Create a connectivity matrix within the same borough
    def get_borough_connectivity_matrix(gdf):
        # Create an empty matrix of connectivity values
        connectivity_matrix = np.zeros((len(gdf), len(gdf)), dtype=int)

        for i, row_i in gdf.iterrows():
            for j, row_j in gdf.iterrows():
                if row_i["borough"] == row_j["borough"] and row_i.geometry.touches(row_j.geometry):
                    connectivity_matrix[i, j] = 1

        return connectivity_matrix

    # Get the connectivity matrix considering boroughs
    connectivity_matrix = get_borough_connectivity_matrix(gdf)

    # Perform hierarchical clustering with borough connectivity constraints
    n_clusters = 130  # Adjust for the number of merged zones
    clustering = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity_matrix, linkage='ward')
    gdf["cluster_id"] = clustering.fit_predict(centroids)

    # Merge geometries based on cluster assignment
    gdf_merged = gdf.dissolve(by="cluster_id").reset_index()
    gdf_merged["zone_id"] = gdf_merged.index + 1  # Assign ordered zone numbers

    # Track the original OBJECTIDs in each merged zone
    def get_original_objectids(cluster_id):
        original_objectids = gdf[gdf['cluster_id'] == cluster_id]['OBJECTID'].tolist()
        return original_objectids

    gdf_merged["original_OBJECTID"] = gdf_merged["cluster_id"].apply(get_original_objectids)

    # Calculate the shape_area and objective (if needed)
    gdf_merged["shape_area"] = gdf_merged.geometry.area
    gdf_merged["objective"] = None  # Replace with actual objective calculation if needed

    # Add the borough column
    gdf_merged["borough"] = gdf_merged["borough"].apply(lambda x: gdf[gdf["borough"] == x]["borough"].iloc[0])

    # Select relevant columns for export
    output_df = gdf_merged[["zone_id", "objective", "shape_area", "borough", "original_OBJECTID"]]

    # Save to CSV
    output_csv_path = "merged_taxi_zones.csv"
    output_df.to_csv(output_csv_path, index=False)
    output_df.to_pickle("merged_taxi_zones.pkl")

    gdf_merged = gdf_merged.drop(columns=["centroid"], errors="ignore")
    output_shp_path = r"shapfile\merged_taxi_zones.shp"
    gdf_merged.to_file(output_shp_path, driver="ESRI Shapefile")

    # Plot the boundaries with numbered zones (for visualization)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_merged.boundary.plot(ax=ax, color="black", linewidth=1)

    # Add zone numbers at centroids
    for idx, row in gdf_merged.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, str(row["zone_id"]), fontsize=8, ha='center', va='center', color="black")

    plt.title("Merged Taxi Zones with Number Labels (Same Boroughs)")
    plt.axis("off")  # Remove axes for a clean look
    # 保存图像为 PDF
    output_pdf_path = "merged_taxi_zones.pdf"
    plt.savefig(output_pdf_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()

if __name__ == "__main__":
    run()