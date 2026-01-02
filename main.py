from data_loader import load_data
from kmeans_model import run_kmeans

def main():
    data = load_data("weather_data.csv")
    labels, model = run_kmeans(data, k=3)

    data['Cluster'] = labels
    print(data.head())

if __name__ == "__main__":
    main()
