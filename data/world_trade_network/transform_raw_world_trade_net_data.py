#########
#Created Date: Friday June 4th 2021
#Author: Domenico Di Gangi,  <digangidomenico@gmail.com>
#-----
#Last Modified: Friday June 4th 2021 11:40:27 pm
#Modified By:  Domenico Di Gangi
#-----
#Description: Downloads the comtrade world trade network dataset and saves it as an artifact
#-----
########

  
import requests
import tempfile
import os
import zipfile
import mlflow
import click


@click.command(
    help="Downloads the world trade network dataset and saves it as an mlflow artifact "
    " called 'world_trade_net_T.npz'."
)
@click.option("--url", default="http://files.grouplens.org/datasets/movielens/ml-20m.zip")
def load_raw_data(url):
    with mlflow.start_run() as mlrun:
        local_dir = tempfile.mkdtemp()
        local_filename = os.path.join(local_dir, "ml-20m.zip")
        print("Downloading %s to %s" % (url, local_filename))
        r = requests.get(url, stream=True)
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        extracted_dir = os.path.join(local_dir, "ml-20m")
        print("Extracting %s into %s" % (local_filename, extracted_dir))
        with zipfile.ZipFile(local_filename, "r") as zip_ref:
            zip_ref.extractall(local_dir)

        ratings_file = os.path.join(extracted_dir, "ratings.csv")

        print("Saving networks : %s" % ratings_file)
        mlflow.log_artifact(ratings_file, "world_trade_net_T.npz")


if __name__ == "__main__":
    load_raw_data()