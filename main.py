"""
main.py

This is the entry point of the K-Means application.
It loads the dataset, applies K-Means clustering,
and displays the clustered results.
"""

# Import function to load the dataset
from data_loader import load_data

# Import function that runs K-Means clustering
from kmeans_model import run_kmeans


def main():
    """
    Main function that controls the program flow.
    """

    # Load weather data from a CSV file
    # The file must exist in the project directory
    data = load_data("weather_data.csv")

    # Apply K-Means clustering with 3 clusters
    labels, model = run_kmeans(data, k=3)

    # Add the cluster labels to the original dataset
    data['Cluster'] = labels

    # Display the first few rows of the clustered data
    print(data.head())


# Ensures the script runs only when executed directly
if __name__ == "__main__":
    main()
