import os
from os import path
import argparse
import wandb
import logging
import pandas as pd
import yaml

from src.hfpipeline_classification import HFPipeline
from src.utils import df_map_softmax_to_columns

import settings
BASE_DIR=settings.BASE_DIR
WANDB_ENTITY=settings.WANDB_ENTITY
WANDB_PROJECT=settings.WANDB_PROJECT
CONFIG_PIPELINE=settings.CONFIG_PIPELINE
STEP_NAME="writters_attitude"

logger = settings.logging.getLogger(__name__)

def get_dataset(run, ds_version:str="latest", **kwargs):
    # get version of the dataset

    artifact_name="ds_detection"
    if run:
        artifact = run.use_artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}:{ds_version}', type='dataset')
    else: # without a run
        api = wandb.Api()
        artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}:{ds_version}")

    artifact_dir = artifact.download()
    return artifact, artifact_dir


def main(log_wandb, localtest, t10sec, **kwargs):
    # TODO test and wrap in a cls and (join?) results and log them to a wandb artifact
    tgt_dir = f"{BASE_DIR}/out"
    logger.info(locals())


    run = None #wandb.init() # provide is art
    if log_wandb:
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, job_type=STEP_NAME) # provide is art

    # load dataset
    if localtest:
        data_path = f"../artifacts/ds_detection:{kwargs.get('ds_version')}/dataset.tsv"
    else:
        # use artifacts
        artifact, local_path = get_dataset(run, **kwargs)
        data_path = os.path.join(local_path, artifact.files()[0].name)

    ds = pd.read_csv(data_path, sep="\t")
    print(ds)

    # run config models and log artifacts
    rs_dss = []
    try:
        # get step config
        with open(CONFIG_PIPELINE, "r") as f:
            config_pipeline = yaml.load(f, Loader=yaml.FullLoader)
        config_step = config_pipeline.get("steps").get(STEP_NAME)

        classification_step = HFPipeline()
        if log_wandb:
            artifact = wandb.Artifact(name=STEP_NAME, type="dataset")
        for config_model in config_step.get("models", []):
            tgt_file = f"{config_model.get('name', config_model.get('model').split('/')[-1])}"
            rs_fpath = classification_step(**config_model, data_path=data_path, tgt_path=path.join(tgt_dir, f"{tgt_file}.tsv"), t10sec=t10sec)
            if log_wandb:
                artifact.add_file(local_path=rs_fpath)

            # try to reformat output files (softmax json -> columns)
            try:
                _ds_formatted = df_map_softmax_to_columns(pd.read_csv(rs_fpath, sep='\t'), 
                            prefix=f"{config_model.get('name','')}_", 
                            label_map=config_model.get('label_map', None))
                _ds_formatted.to_csv(path.join(tgt_dir, f"{tgt_file}_fmt.tsv"), sep="\t")
                rs_dss.append( _ds_formatted )
                if log_wandb:
                    artifact.add_file(local_path=path.join(tgt_dir, f"{tgt_file}_fmt.tsv")) # formatted file
            except Exception as err1:
                logger.warning(f"Errors on formatting results [{tgt_file}]")
                pass
            # --/
            

        # try to merge all predictions
        try:
            _df_merged = pd.concat(rs_dss, axis=1)
            _df_merged.to_csv(path.join(tgt_dir, "merged_fmt.tsv"), sep='\t')
            if log_wandb:
                artifact.add_file(local_path=path.join(tgt_dir, "merged_fmt.tsv"))
        except Exception as err2:
            logger.warning(f"Errors on merging formatted files")
            pass            
        # --/

        if log_wandb:
            run.log_artifact(artifact)
            run.finish()

    except Exception as err:
        logger.error("\n{}\nERROR on accessing step config".format('-'*30), exc_info=err)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument("--ds_version", default="latest", type=str, help="dataset version")
    parser.add_argument("--log_wandb", action="store_true", help="log to wandb")
    parser.add_argument("--localtest", action="store_true", help="ONLY use for local unit tests")
    parser.add_argument('--t10sec', type=bool, default=False, help="Sanity check (Unitest)")

    args = parser.parse_args()

    main(**vars(args))
