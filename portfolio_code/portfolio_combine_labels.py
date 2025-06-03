# merge.py

import os
import pandas as pd

def main():
    # File paths
    selected_data_path = 'selected_data_completed.csv'
    nn3_path           = 'predicted_weighted_test_label3.csv'
    nn7_path           = 'predicted_weighted_testing_label7.csv'

    # File verification
    for p in (selected_data_path, nn3_path, nn7_path):
        if not os.path.isfile(p):
            raise SystemExit(f"Error: '{p}' not found. Place it alongside this script.")

    # Load and parse date
    df_sel = pd.read_csv(selected_data_path, parse_dates=['date'])

    # Load test3 and extract only (stock, date, predicted_label_3)
    df_nn3 = pd.read_csv(nn3_path, parse_dates=['date'])
    df_nn3 = df_nn3[['stock', 'date', 'nn3']]

    # Load test7 and extract only (stock, date, predicted_label_7)
    df_nn7 = pd.read_csv(nn7_path, parse_dates=['date'])
    df_nn7 = df_nn7[['stock', 'date', 'nn7']]

    # Merge nn3 labels
    df_merged = pd.merge(
        df_sel,
        df_nn3,
        on=['stock', 'date'],
        how='left'  # use left join so we keep all original rows (even if NN has no label)
    )

    # Merge nn7 labels
    df_merged = pd.merge(
        df_merged,
        df_nn7,
        on=['stock', 'date'],
        how='left'
    )

    # Save to new csv
    output_path = 'selected_data_with_nn.csv'
    df_merged.to_csv(output_path, index=False)
    print(f"Merged file written to '{output_path}'")

if __name__ == "__main__":
    main()
