#! Place each genration or human wirtten contibution in a separate ro
import pandas as pd
import uuid


with open('../configs/configs.yaml') as f:
        configs = yaml.safe_load(f)

org_data_file_name = "[THE ORIGINAL MELTED DATASET FILE NAME]" # e.g. "databricks-dolly_size-2000_benchmark_MELTED"

in_file_name = "[THE INPUT DATASET FILE NAME]" 


def get_data_id(original_df, instruction=None, context=None, category=None, response=None):
    # check if context is nan
    if pd.isna(context):
        df_tmp = original_df[(original_df['instruction'] == instruction) & (original_df['context'].isna()) & (original_df['category'] == category) & (original_df['response'] == response)]
    else:
        df_tmp = original_df[(original_df['instruction'] == instruction) & (original_df['context'] == context) & (original_df['category'] == category) & (original_df['response'] == response)]
    # Sanity check that there is only one row
    assert len(df_tmp.index) == 1, f"Contains more than one row: {df_tmp}"
    return df_tmp['data_id'].iloc[0]


df = pd.read_csv(f'../results/{in_file_name}.csv')

# Abaltion study
if configs['is_ablation_length'] or configs['is_ablation_temperature']:
    print("****** ABLATION STUDY ******")
    print("----- DATA ID WILL BE GENERATED FROM ORIGINAL DATASET -----")
    
    df_original = pd.read_csv(f'../results/{org_data_file_name}.csv')
    df_original = df_original[df_original['model'] == 'human'] 
    df['data_id'] = df.apply(lambda row: get_data_id(df_original, instruction=row['instruction'], context=row['context'], category=row['category'], response=row['response']), axis=1)

else:
    df["data_id"] = [uuid.uuid4().hex for _ in range(len(df.index))]

print(df.head())
print(df.columns)
print(df.shape)

df.rename(columns={'response': 'human'}, inplace=True)
df_melted = df.melt(id_vars=['instruction', 'context', 'category', 'data_id'], var_name='model', value_name='response')

df_melted["id"] = [uuid.uuid4().hex for _ in range(len(df_melted.index))]

print(df_melted.head(30))
print(df_melted.columns)
print(df_melted.shape)
print(df_melted['model'].value_counts())

df_melted.to_csv(f'../results/{in_file_name}_melted.csv', index=False, escapechar='\\')
df_melted.to_csv(f'../results/{in_file_name}_melted.tsv', index=False, sep="\t", escapechar='\\')