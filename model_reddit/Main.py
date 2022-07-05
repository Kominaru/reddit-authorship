# -*- coding: utf-8 -*-

import argparse
import nvgpu

from src.ImgModel import *
from codecarbon import EmissionsTracker

########################################################################################################################


def cmd_read_args():
    """
    Read commandline args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, help="Modelo a utilizar")
    parser.add_argument('-stage', type=str, help="grid o test")
    parser.add_argument('-gpu', type=str, help="Gpu")
    parser.add_argument('-lr', nargs='+', type=float, help='Lista de learning-rates a probar')
    parser.add_argument('-d', nargs='+', type=float, help="DropOut")
    parser.add_argument('-nimg', type=str, help="Negativos para las imágenes (5,5+5...[dentro del rest y fuera])", )
    parser.add_argument('-s', type=int, help="Semilla")
    parser.add_argument('-e', type=int, help="Epochs")
    parser.add_argument('-c', type=str, help="Ciudad", )

    ret_args = parser.parse_args()

    return ret_args


########################################################################################################################


args = cmd_read_args()

stage = "test" if args.stage is None else args.stage

gpu = np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))) if args.gpu is None else args.gpu

lrates = [5e-4] if args.lr is None else args.lr
dpouts = [0.2] if args.d is None else args.d
nimg = "10+10" if args.nimg is None else args.nimg
epochs = 25 if args.e is None else args.e
seed = 100 if args.s is None else args.s
city = "gijon" if args.c is None else args.c

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

config = {
        "neg_images": nimg,
        "learning_rate": lrates[0],
        "dropout": dpouts[0],
        "epochs": epochs,
        "batch_size": 2 ** 13  # Our GPU can handle this size
        }

########################################################################################################################

model = ImgModel(city=city, config=config, seed=seed)

if "stats" in stage:
    model.get_data_stats()

if "grid" in stage:
    params = {"learning_rate": lrates, "dropout": dpouts}
    model.grid_search(params, max_epochs=epochs, start_n_epochs=epochs)

if "train" in stage:
    tracker=EmissionsTracker(log_level='error',measure_power_secs=1200)
    tracker.start()
        
    model.final_train(epochs=50, save=True)

    emissions=tracker.stop()
    print(f"Total emissions are {emissions} kg")

if "test" in stage:
        # model.final_train(epochs=epochs, save=True)
        model.get_detailed_results()