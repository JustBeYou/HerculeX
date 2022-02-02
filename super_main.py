"""

Training pipeline

0. If there are no existing models, generate two models, select the winning one
1. choose the best model and start self-play on N threads (batch of M games)
2. train the best model on K random batches of games
3. evaluate candidates on Q*2 matches and replace the best model if winning percentage is >P (half red/half blue)

"""
import constants

N = 2
M = 200
K = 5
Q = 75
B = 75
P = 0.55

from multiprocessing import Process
from os import listdir, unlink
import os
from shutil import move
from random import randint, seed
from time import sleep

import os
import binascii

SEED = int(binascii.hexlify(os.urandom(8)), 16)
seed(SEED)

cand_path = "./pipeline_data/candidate_models"
best_path = "./pipeline_data/best_model"
exp_path = "./pipeline_data/history"
old_path = "./pipeline_data/old"
save_all = f"--save-agent-path {cand_path} --save-opponent-path {cand_path}"
load_same = lambda p: f"--load-agent-path {p} --load-opponent-path {p}"
herculex = "--agent HerculexTheSecond --opponent HerculexTheSecond"

def parse_id(fname):
    id = fname.split(".")[-2]
    parts = id.split("_")
    return id, parts

def get_name(fname):
    return fname.split(".")[-3]

def get_rand_seed():
    return randint(0, int(1e8))

real_main_func = None

def do(cmd):
    print(f"[i] CMD: {cmd}")
    return real_main_func(cmd.split())
    #return {}

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

    print(f"[+] Base model {listdir(best_path)[0]}")

import gc

def self_play(i):
    try:
        best = listdir(best_path)[0]

        print(f"[i] Play task {i} Model {best}. Will play {M} games.")

        best_model_path = f"{best_path}/{best}"
        info = do(f'run {herculex} --episodes {M} {load_same(best_model_path)} --save-experience-path {exp_path}')

        ### DEBUG
        #id, _ = parse_id(best)
        #open(f"./pipeline_data/history/experience.{id}_{get_rand_seed()}.npz", "w").write("debug")
        #sleep(2)
        ###

        print(f"[+] Play task {i} Model {best}. Summary: {info}")
    except Exception as e:
        print(f"[!] Play task {i} failed. Exception:", e)
        sleep(5)

    # TODO: delete this exit!!!
    #exit()
    gc.collect()

def train():
    try:
        best = listdir(best_path)[0]

        cands_len = len(listdir(cand_path))
        if cands_len > 3:
            print(f"[i] Train task sleep. Too many candidates.")
            sleep(15)
            return

        print(f"[i] Train task. Model {best}. Will train {K} batches of games.")

        best_model_path = f"{best_path}/{best}"
        id, parts = parse_id(best)
        version, _ = parts
        new_candidate_name = f"{get_name(best)}.{int(version) + 1}_{get_rand_seed()}.h5"
        new_candidate_path = f"{cand_path}/{new_candidate_name}"

        did_train = do(f"train --experience-sample {K} --load-agent-path {best_model_path} --save-agent-path {new_candidate_path}")

        if not did_train:
            print("[-] Nothing to train. Will sleep.")
            sleep(15)
        else:
            print(f"[+] Train task. Model {best}. New candidate {new_candidate_name}")
            for i in range(constants.TRAIN_ITERS):
                do(f"train --experience-sample {K} --load-agent-path {best_model_path} --save-agent-path {new_candidate_path}")

        ### DEBUG
        #open(new_candidate_path, "w").write("debug")
        #sleep(3)
        ###
    except Exception as e:
        print(f"[!] Train task failed. Exception:", e)
        sleep(5)

def evaluate():
    try:
        candidates = listdir(cand_path)

        matches = 2*Q
        print(f"[i] Evaluate task. Will evaluate {len(candidates)} candidates, each on {matches} matches.")

        if len(candidates) == 0:
            print("[-] Nothing to evaluate. Will sleep")
            sleep(15)
            return

        for candidate in candidates:
            best = listdir(best_path)[0]
            print(f"[i] Evaluating candidate {candidate} vs best {best}")

            best_model_path = f"{best_path}/{best}"
            candidate_model_path = f"{cand_path}/{candidate}"

            cand_id = parse_id(candidate)[0]
            best_id = parse_id(best)[0]

            ### DEBUG
            #cand_wins = randint(0, matches)
            #results = {
            #    "wins": {
            #        cand_id: cand_wins,
            #        best_id: matches - cand_wins
            #    }
            #}
            #sleep(2)
            ###

            results1 = do(f'run {herculex} --episodes {Q} --load-agent-path {best_model_path} --load-opponent-path {candidate_model_path}')
            results2 = do(f'run {herculex} --episodes {Q} --load-agent-path {candidate_model_path} --load-opponent-path {best_model_path}')

            results = {
                "wins": {
                    cand_id: results1["wins"][cand_id] + results2["wins"][cand_id],
                    best_id: results1["wins"][best_id] + results2["wins"][best_id],
                }
            }

            win_rate = results["wins"][cand_id] / matches
            print(f"[+] Evaluate task. Candidate {candidate} vs best {best}. Win rate candidate: {win_rate}")

            if win_rate >= P:
                print(f"[+] Evaluate task. Candidate {candidate} wins!")
                move(candidate_model_path, best_path)
                move(best_model_path, old_path)
            else:
                print(f"[+] Evaluate task. Best {best} wins!")
                unlink(candidate_model_path)
    except Exception as e:
        print(f"[!] Evaluate task failed. Exception:", e)
        sleep(5)

def benchmark():
    try:
        best = listdir(best_path)[0]
        best_model_path = f"{best_path}/{best}"
        best_id = parse_id(best)[0]

        print(f"[i] Benchmark task. {best} on {2*B} matches.")

        results1 = do(
           f'run --agent HerculexTheSecond --opponent RandomAgent --episodes {B} --load-agent-path {best_model_path}')
        results2 = do(
            f'run  --agent RandomAgent --opponent HerculexTheSecond --episodes {B} --load-opponent-path {best_model_path}')

        results = {
            "wins": {
                "randomvirgin": results1["wins"]["randomvirgin"] + results2["wins"]["randomvirgin"],
                best_id: results1["wins"][best_id] + results2["wins"][best_id],
            }
        }

        matches = 2*B
        win_rate = results["wins"][best_id] / matches
        print(f"[+] Benchmark task. Best {best} vs random agent. Win rate best: {win_rate}")
        sleep(60)
    except Exception as e:
        print(f"[!] Benchmark task failed. Exception:", e)
        sleep(60)

def forever(func, *args):
    global real_main_func
    from main import main as real_main
    real_main_func = real_main
    while True:
        func(*args)
        sleep(1)

def main():
    pass

if __name__ == '__main__':
    dirs = ["./pipeline_data", cand_path, best_path, exp_path, old_path]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    if len(listdir('./pipeline_data/best_model')) == 0:
        prepare()

    processes = []
    for i in range(N):
        processes.append(Process(target=forever, args=(self_play, i)))
        processes[-1].start()

    processes.append(Process(target=forever, args=(train,)))
    processes[-1].start()

    processes.append(Process(target=forever, args=(evaluate,)))
    processes[-1].start()

    processes.append(Process(target=forever, args=(benchmark,)))
    processes[-1].start()

    for p in processes:
        p.join()