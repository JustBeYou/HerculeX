"""

Training pipeline

0. If there are no existing models, generate two models, select the winning one
1. choose the best model and start self-play on N threads (batch of M games)
2. train the best model on K random batches of games
3. evaluate candidates and replace the best model if winning percentage is >P (half red/half blue)

"""

N = 2
M = 1
K = 3
P = 0.55

from multiprocessing import Pool
from os import listdir, unlink, system
from main import main as real_main
from shutil import move

cand_path = "./pipeline_data/candidate_models"
best_path = "./pipeline_data/best_model"
exp_path = "./pipeline_data/history"
save_all = f"--save-agent-path {cand_path} --save-opponent-path {cand_path}"
load_same = lambda p: f"--load-agent-path {p} --load-opponent-path {p}"
herculex = "--agent HerculexTheSecond --opponent HerculexTheSecond"

def do(cmd):
    print(f"[i] CMD: {cmd}")
    return real_main(cmd.split())

def prepare():
    print("[i] Preparing base model...")

    info = do(f'run {herculex} --episodes 1 {save_all}')
    ids = list(info["wins"].keys())

    if info["wins"][ids[0]] == 1:
        winner = ids[0]
    else:
        winner = ids[1]

    for fname in listdir(cand_path):
        if winner in fname:
            move(f"{cand_path}/{fname}", best_path)
        else:
            unlink(f"{cand_path}/{fname}")

    print(f"[i] Base model {listdir(best_path)[0]}")

def self_play(data):
    i, best = data

    print(f"[i] Play task {i} Model {best}. Will play {M} games.")

    best_model_path = f"{best_path}/{best}"
    info = do(f'run {herculex} --episodes {M} {load_same(best_model_path)} --save-experience-path {exp_path}')

    print(f"[i] Play task {i} Model {best}. Summary: {info}")

def train():
    pass

def evaluate():
    pass

def main():
    pass

if __name__ == '__main__':
    if len(listdir('./pipeline_data/best_model')) == 0:
        prepare()

    self_play((0, "residual.0_75788177.h5"))