import ast
import pandas as pd

def df_map_softmax_to_columns(df:pd.DataFrame, label_map:dict=None, prefix:str=None) -> pd.DataFrame:
    """Map classification softmax objects to columns

    Args:
        df (pd.DataFrame): dataframe with softmaxs
        label_map (dict, optional): Dict {old_name: new_name} to rename columns. Defaults to None.
        prefix (str, optional): prefix to be added to each column (e.g 'mytype_'). Defaults to None.

    Returns:
        Dataframe: modified dataframe        
    """

    _labels = ( [ ast.literal_eval(v)['label'] for v  in df.iloc[0,:].values if type(v)==str ])
    df = df.iloc[:,1:].rename(lambda n: f"{_labels[int(n)]}", axis="columns")
    df = df.applymap(lambda e: ast.literal_eval(e)['score'])
    if label_map:
        df.rename(columns=label_map, inplace=True)
    if prefix:
        df.rename(columns={c: f"{prefix}{c}" for c in df.columns}, inplace=True)
        
    return df