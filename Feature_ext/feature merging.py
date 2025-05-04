import pandas as pd

colon_aca_model_csv_path = "/content/drive/MyDrive/bitirmeprojesi/sdata/combined_features/colon_aca_features2.csv"
colon_aca_lbp_csv_path = "/content/drive/MyDrive/bitirmeprojesi/sdata/lbp/combined_features/colon_aca_lbp_features.csv"
colon_aca_output_csv_path = "/content/drive/MyDrive/bitirmeprojesi/sdata/combined_features/combined_model_lbp_colon_aca.csv"

colon_n_model_csv_path = "/content/drive/MyDrive/bitirmeprojesi/sdata/combined_features/colon_n_features2.csv"
colon_n_lbp_csv_path = "/content/drive/MyDrive/bitirmeprojesi/sdata/lbp/combined_features/colon_n_lbp_features.csv"
colon_n_output_csv_path = "/content/drive/MyDrive/bitirmeprojesi/sdata/combined_features/combined_model_lbp_colon_n.csv"

df_colon_aca_model = pd.read_csv(colon_aca_model_csv_path)
df_colon_aca_lbp = pd.read_csv(colon_aca_lbp_csv_path)

df_colon_n_model = pd.read_csv(colon_n_model_csv_path)
df_colon_n_lbp = pd.read_csv(colon_n_lbp_csv_path)

all_columns_aca = sorted(set(df_colon_aca_model.columns).union(df_colon_aca_lbp.columns))
df_colon_aca_model = df_colon_aca_model.reindex(columns=all_columns_aca)
df_colon_aca_lbp = df_colon_aca_lbp.reindex(columns=all_columns_aca)

all_columns_n = sorted(set(df_colon_n_model.columns).union(df_colon_n_lbp.columns))
df_colon_n_model = df_colon_n_model.reindex(columns=all_columns_n)
df_colon_n_lbp = df_colon_n_lbp.reindex(columns=all_columns_n)

df_colon_aca_combined = pd.concat([df_colon_aca_model, df_colon_aca_lbp], ignore_index=True).fillna('')
df_colon_n_combined = pd.concat([df_colon_n_model, df_colon_n_lbp], ignore_index=True).fillna('')


df_colon_aca_combined.to_csv(colon_aca_output_csv_path, index=False)
df_colon_n_combined.to_csv(colon_n_output_csv_path, index=False)


