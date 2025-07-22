# action.py
from analysis import load_data, complaints_by_date, heatmap_by_borough

if __name__ == "__main__":
    df = load_data("311_Service_Requests_from_2010_to_Present_20250718.csv")
    complaints_by_date(df)
    heatmap_by_borough(df)
