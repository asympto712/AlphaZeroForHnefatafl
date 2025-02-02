import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from collections import deque
from typing import Literal

from .utils import *
from .taflNNet import TaflNNet
from ._azhnefatafl import self_play_function

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'maxlen': 10000,
    'numIter': 10,
    'numGamesPerIter': 10,
}

game = {
    'boardsize' : (7, 7),
    'actionsize' : 49 * 49,
    }


class NNetWrapper():
    def __init__(self, args, game, jit_model=None):
        if jit_model is not None:
            self.nnet = jit_model
            if args["cuda"]:
                self.nnet = jit_model.to('cuda')
        self.args = args
        self.game = game
        self.latest_checkpoint_path = None
        self.latest_train_examples_path = None

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, player, v)
        board should already be a matrix representation
        v should always be from the perspective of the attacker 
        """
        optimizer = optim.Adam(self.nnet.parameters())
        if not self.nnet:
            print("this wrapper instance has no attribute nnet. Load an already existing model and try again.")
            return None
        elif self.args['cuda']:
            self.nnet.to('cuda')
        
        for epoch in range(self.args["epochs"]):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()

            batch_count = int(len(examples) / self.args["batch_size"])

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args["batch_size"])

                # If the examples are given as a structured numpy array. That is, if it was loaded from .npz file
                if isinstance(examples, np.ndarray):
                    boards = torch.FloatTensor(examples["boards"])
                    target_pis = torch.FloatTensor(examples["pis"])
                    players = torch.BoolTensor(examples["players"])
                    target_vs = torch.FloatTensor(examples["vs"])
                else:
                    boards, pis, players, vs = list(zip(*[examples[i] for i in sample_ids]))
                    # boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    boards = torch.FloatTensor(np.array(boards))
                    target_pis = torch.FloatTensor(np.array(pis))
                    players = torch.BoolTensor([True if player == 1 else False for player in players])
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if self.args["cuda"]:
                    boards, target_pis, players, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), players.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                pis, vs = self.nnet(boards, players)
                
                l_pi = self.loss_pi(target_pis, pis)
                l_v = self.loss_v(target_vs, vs)
                total_loss = l_pi + l_v

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    # def predict(self, board):  # remnant from Alpha-Zero-General. We won't be using this.
    #     """
    #     board: np array with board
    #     """
    #     # timing
    #     start = time.time()

    #     # preparing input
    #     board = torch.FloatTensor(board.astype(np.float64))
    #     if args["cuda"]: board = board.contiguous().cuda()
    #     board = board.view(1, self.board_x, self.board_y)
    #     self.nnet.eval()
    #     with torch.no_grad():
    #         pi, v = self.nnet(board)

    #     # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
    #     return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    # output is log_softmax
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


    def save_checkpoint(self, saveasexample: bool, prefix: Literal['i','c'], folder='models', filename='example.pt'):
        """
        this will save jit.scripted model into a filepath dependent on the current time( e.g. chp_0951_01.02.25.pt)
        & return that filepath as the output.
        prefix 'i' indicates that the model was initialized. 'c' indicates that it is a successor to another checkpoint
        """
        if not saveasexample:
            cur_time = time.strftime("%H%M_%d.%m.%y")
            filename = prefix + '_' + cur_time + '.pt'

        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        
        self.nnet.save(filepath)
        print("scripted model saved at: {}".format(filepath))        
        return filepath

    def load_checkpoint(self, filepath): 
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        scripted_model = torch.jit.load(filepath)
        self.nnet = scripted_model
        if self.args['cuda']:
            self.nnet = scripted_model.to('cuda')

    def generate_train_examples(self, model_path, old_train_examples = None, save = True):
        if old_train_examples: # Pass a deque object with maxlen specified
            train_examples = old_train_examples
        else:
            train_examples = deque([], maxlen= self.args['maxlen'])
        
        t = tqdm(range(self.args['numIter']), desc="generating train examples")
        for _ in t:
            t.set_description(f"Iteration: {_ + 1}")
            # Generate new training examples using self-play
            new_examples = self_play_function(model_path, self.args['numGamesPerIter'])
            # add the new_examples to the right side of the deque (if length exceeds the maxlen, it will discard older examples from the left)
            train_examples.extend(new_examples)

        reformatted = []
        for example in train_examples:
            np_matrix = np.array([np.frombuffer(row, dtype=np.uint8) for row in example[0]])
            tmp = (np_matrix,example[1],example[2],example[3])
            reformatted.append(tmp)    


        # if save, save the structured numpy array from the train_examples to train_examples/ folder
        if save:
            filepath = self.save_train_examples(reformatted)
            self.latest_train_examples_path = filepath

        
        return reformatted
    
    def save_train_examples(self, train_examples):
        cur_time = time.strftime("%H%M_%d.%m.%y")
        filename = cur_time + '.npz'
        filepath = os.path.join("train_examples", filename)
        if not os.path.exists("train_examples"):
            print("Train examples directory does not exist! Making directory train_examples")
            os.mkdir("train_examples")

        dtypes = np.dtype([
            ("boards", np.uint8, self.game['boardsize']),
            ("pis", np.float32, (self.game['actionsize'])),
            ("players", np.int8),
            ("vs", np.float32)
        ])

        np_array = np.array(train_examples, dtype=dtypes)
        np.savez_compressed(filepath, a=np_array)
        print("train_examples saved at {}".format(filepath))
        return filepath

    def load_train_examples(self, path, use: Literal["train", "generate"]):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path}")
        loaded = np.load(path)
        
        if use == "train":
            return loaded['a']
        else:
            return deque((item.item() for item in loaded['a']), maxlen=self.args['maxlen'])

    def learn(self, checkpoint_filepath = None, train_examples_path = None):
        """
        About checkpoint_filepath:
        If checkpoint_filepath is given, use that model and set it as the latest checkpoint.
        If not, first it checks if self has a latest_checkpoint_path attribute. If it does, use the model indicated by that path.
        If not, it creates an initialized model and set it as the latest checkpoint.

        About train_examples_path:
        Similar system as checkpoint_filepath
        """

        count = 1
        try:
            while True:

                print("Starting the virtuous train cycle {}!".format(count))
                if checkpoint_filepath:
                    self.latest_checkpoint_path = checkpoint_filepath
                elif self.latest_checkpoint_path:
                    print("No checkpoint_filepath argument was given, using model at {}".format(self.latest_checkpoint_path))
                    time.sleep(1)
                    
                else:
                    print("No checkpoint_filepath was given & no information on the latest checkpoint. Creating a new model..")
                    time.sleep(1)
                    initial_model = TaflNNet(self.game, self.args)
                    scripted_initial_model = torch.jit.script(initial_model)
                    self.nnet = scripted_initial_model
                    if self.args['cuda']:
                        self.nnet = scripted_initial_model.to('cuda')
                    path = self.save_checkpoint(saveasexample=False, prefix='i')
                    self.latest_checkpoint_path = path
                    print("New model was created and saved at {}".format(path))
                    time.sleep(1)


                if train_examples_path:
                    self.latest_train_examples_path = train_examples_path
                    old_train_examples = self.load_train_examples(self.latest_train_examples_path, 'generate')
                elif self.latest_train_examples_path:
                    print("No train_example_path was given. Using the latest train_example..")
                    time.sleep(1)
                    old_train_examples = self.load_train_examples(self.latest_train_examples_path, 'generate')
                else:
                    print("No train_exmaple_path was given nor is there any infomation on the latest train_examples. Creating new train examples")
                    time.sleep(1)
                    old_train_examples = None

                # PRINTING FOR DEBUGGING
                train_examples = self.generate_train_examples(self.latest_checkpoint_path, old_train_examples, save=True)
                # print("Last 5 examples: \n", train_examples[-5:])

                self.load_checkpoint(self.latest_checkpoint_path)
                self.train(train_examples)

                del train_examples

                model_path = self.save_checkpoint(saveasexample=False, prefix='c')
                self.latest_checkpoint_path = model_path

                count += 1

        except KeyboardInterrupt:
            print("Training interrupted by user. latest checkpoints are.. \nmodel:{} \ntraining_examples:{}"
                  .format(self.latest_checkpoint_path, self.latest_train_examples_path))
            
    def save_itself(self, savennmodelaswell = False):
        if not savennmodelaswell:
            self.nnet = None
        with open("wrapper.pkl", "wb") as f:
            pickle.dump(self, f)
            print("wrapper saved! To load, \n"
                  + "with open(\"wrapper.pkl\", \"rb\") as f:\n"
                  + "   loaded_wrapper = pickle.load(f)")
        



