# git clone https://github.com/EvelynBunnyDev/S5.git

# pip install flax
# pip install torch
# pip install torchtext
# pip install tensorflow-datasets==4.5.2
# pip install pydub==0.25.1
# pip install datasets
# pip install tqdm

# pip install wandb
# pip install einops

import jax
import jax.numpy as jnp
from jax import random, grad, jit
from jax.scipy.linalg import block_diag
import numpy as np
import numpy.random as npr
import os
import pickle

import importlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from flax import linen as nn
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from tqdm import tqdm
import optax
import wandb

# Initializing the S5 model
from s5.seq_model import AutoencoderModel, BatchAutoencoderModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5 import dataloading
from s5 import seq_model
from s5 import train_helpers

# parameters we want to track in wandb
default_config = dict(
    ssm_size = 256,
    ssm_lr = 1e-3,
    blocks = 8,
    lr_factor = 1.0,
    # Model Hyperparameters
    d_model = 256,
    n_layers = 3,
    d_latent=3, # bottleneck dimension
    d_output = 216, # output dim for ssm layer
    data_seq_len = 100, # input sequence length
    ratio = 0.85, # Training proportion
    bsz = 32,
    # Timing (task) parameters
    dt_min = 0.001,
    dt_max = 0.1,
    # Training
    num_iters = 200,
    learning_rate = 1e-3,
    keep_rate = 0.95,
    key = 13,
)

projectname = "s5_autoencoder"
wandb.init(config=default_config, project=projectname)
config = wandb.config

def log_wandb_model(model, name, type):
    trained_model_artifact = wandb.Artifact(name,type=type)
    if not os.path.isdir('models'): os.mkdir('models')
    subdirectory = wandb.run.name
    filepath = os.path.join('models', subdirectory)
    try: os.mkdir(filepath)
    except: filepath=filepath
    obs_outfile = open(os.path.join(filepath, "model"), 'wb')
    pickle.dump(model, obs_outfile)
    obs_outfile.close()
    trained_model_artifact.add_dir(filepath)
    wandb.log_artifact(trained_model_artifact)

# determine the size of initial blocks
block_size = int(config['ssm_size'] / config['blocks'])

# Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
lr = config['lr_factor'] * config['ssm_lr']

# Set randomness...
print("[*] Setting Randomness...")
key = random.PRNGKey(config['key'])
init_rng, train_rng = random.split(key, num=2)

padded = False
retrieval = False

# Initialize state matrix A using approximation to HiPPO-LegS matrix
Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

conj_sym = True
if conj_sym:
    block_size = block_size // 2
    ssm_size = config['ssm_size'] // 2
else:
    ssm_size = config['ssm_size']
    
Lambda = Lambda[:block_size]
V = V[:, :block_size]
Vc = V.conj().T

# If initializing state matrix A as block-diagonal, put HiPPO approximation
# on each block
Lambda = (Lambda * jnp.ones((config['blocks'], block_size))).ravel()
V = block_diag(*([V] * config['blocks']))
Vinv = block_diag(*([Vc] * config['blocks']))

print("Lambda.shape={}".format(Lambda.shape))
print("V.shape={}".format(V.shape))
print("Vinv.shape={}".format(Vinv.shape))

d_model = config['d_model'] # input dim
C_init = 'trunc_standard_normal'
discretization = 'zoh'
dt_min = config['dt_min']
dt_max = config['dt_max']
clip_eigs = False
bidirectional = False

ssm_init_fn = init_S5SSM(H=d_model,
                          P=ssm_size,
                          Lambda_re_init=Lambda.real,
                          Lambda_im_init=Lambda.imag,
                          V=V,
                          Vinv=Vinv,
                          C_init=C_init,
                          discretization=discretization,
                          dt_min=dt_min,
                          dt_max=dt_max,
                          conj_sym=conj_sym,
                          clip_eigs=clip_eigs,
                          bidirectional=bidirectional)

BatchAutoencoderModel = nn.vmap(
    AutoencoderModel,
    in_axes=(0, 0, None),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')

n_layers = config['n_layers']
activation_fn = 'half_glu1'
p_dropout = 0.0 # Range 0.0 - 1.0 for dropout proportions
prenorm= False
batchnorm= True
bn_momentum= 0.95
d_output = config['d_output']
d_latent = config['d_latent']

model_cls = partial(
    BatchAutoencoderModel,
    ssm=ssm_init_fn,
    d_model=d_model,
    d_latent=d_latent,
    n_layers=n_layers,
    activation=activation_fn,
    dropout=p_dropout,
    prenorm=prenorm,
    batchnorm=batchnorm,
    bn_momentum=bn_momentum,
    d_output=d_output,
    padded=False
)

# using poisson loss
# tfd.Poisson(rate=5).sample(seed=key) # sampling one value from a poisson distribution with rate 5
# tfd.Poisson(rate=5).log_prob(1) # log probability of observing 1 in this distribution

def objective(params, vars, inputs, integration_timesteps, targets, dropout_rng):
    (outputs, latents), vars = model.apply(
        {"params": params, "batch_stats": vars["batch_stats"]},
        inputs, integration_timesteps, True,
        rngs={"dropout": dropout_rng},
        mutable=["intermediates", "batch_stats"],
    )
    batch_size = targets.shape[0]
    print(outputs.shape, targets.shape)

    loss = -1.0 * tfd.Poisson(outputs).log_prob(targets).sum() / batch_size
    vars["loss"] = loss
    return loss, vars

# Data Loading
data_seq_len = config['data_seq_len']

data = np.load('/scratch/users/evsong/Low_Pisa_0430_Processed_Full.npz')
print("Available arrays:", data.files)
print(data['reward'].shape, data['ys'].shape, data['xs'].shape) # ys = 357087 time steps, 216 features
data_y = data['ys']

class SpikesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return ((len(self.data) + 1) // data_seq_len) - 1

    def __getitem__(self, idx):
        # Split ys into chunks with data_seq_len
        return self.data[idx * data_seq_len : (idx + 1) * data_seq_len]

ratio = config['ratio']
dataset = SpikesDataset(data_y)
train_sz = int(ratio * len(dataset))
test_sz = len(dataset) - train_sz

# Create dataset instances
train_dataset, test_dataset = random_split(dataset, [train_sz, test_sz])

bsz = config['bsz']
ys_train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, drop_last=True)
ys_test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False, drop_last=True)

# plt.plot(xs[0])

# plt.plot(lambdas_train[0])

# print(xs.shape)
# print(ys.shape)

seq_len = data_seq_len
in_dim = 216

init_rng, dropout_rng = random.split(init_rng, num=2)
model = model_cls(training=True)
integration_timesteps_x = jnp.ones((bsz, seq_len,))

# params = variables['params']
# batch_stats = variables["batch_stats"]
# key, skey = random.split(key)

# outputs, vars = model.apply(
#     {"params": params, "batch_stats": batch_stats},
#     ys_batch, integration_timesteps_x, True,
#     rngs={"dropout": dropout_rng},
#     mutable=["intermediates", "batch_stats"],
# )

# define test set to be first bsz test trials
# ys_batch_test = ys_test[:bsz]

# eval_loss, _ = objective(params, vars, ys_batch_test, integration_timesteps_x, ys_batch_test, skey) # Making sure one iter can run

@jit
def dropout(x, key, keep_rate):
    do_keep = random.bernoulli(key, keep_rate, x.shape)
    kept_rates = jnp.where(do_keep, x / keep_rate, 0.0)
    return jnp.where(keep_rate < 1.0, kept_rates, x)

grad_obj = jit(grad(objective, has_aux=True))
objective = jit(objective)

learning_rate = config['learning_rate']
optimizer = optax.adam(learning_rate)

num_iters = config['num_iters']
losses = []
eval_losses = []

best_loss = np.inf
best_params = None

keep_rate = config['keep_rate']

train_len = len(ys_train_loader)
test_len = len(ys_test_loader)

for i in tqdm(range(num_iters)):
    avg_train_loss = 0
    avg_eval_loss = 0
    batch_cnt = 0

    print("\nTrain:")
    for ys_in_batch in tqdm(ys_train_loader, total=train_len):
        ys_in_batch = jnp.array(ys_in_batch.numpy())
        # initialize
        if i == 0 and batch_cnt == 0:
            # print(ys_in_batch.shape, integration_timesteps_x.shape)
            vars = model.init({"params":init_rng, "dropout": dropout_rng}, ys_in_batch, integration_timesteps_x, True)
            params = vars['params']
            batch_stats = vars["batch_stats"]
            key, skey = random.split(key)

            opt_state = optimizer.init(params)

        # sample some data
        key, skey = random.split(key)

        # take gradient of objective
        key, *skeys = random.split(key, 3)
        ys_in_processed = dropout(ys_in_batch, skeys[0], keep_rate)
        grads, vars = grad_obj(params, vars, ys_in_processed, integration_timesteps_x, ys_in_batch, skeys[1])

        # update parameters based on gradient
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        loss = vars['loss']
        wandb.log({"loss": loss})
        print("\nLoss:", loss)
        avg_train_loss += loss / train_len

        batch_cnt += 1

    print("\nEval:")
    for ys_in_batch_eval in tqdm(ys_test_loader, total=test_len):
        ys_in_batch_eval = jnp.array(ys_in_batch_eval.numpy())
        # compute eval loss
        eval_loss, _ = objective(params, vars, ys_in_batch_eval, integration_timesteps_x, ys_in_batch_eval, skey)
    
        avg_eval_loss += eval_loss / test_len

    if avg_eval_loss < best_loss:
        best_loss = eval_loss
        best_params = params.copy()
        
        log_wandb_model(best_params, "autoencoder_{}_best".format(config['data_seq_len'], config['learning_rate']), 'model')

        
    print(f"\nEpoch {i} || Train Loss: {avg_train_loss:.4f} || Eval Loss: {avg_eval_loss:.4f}")

    losses.append(avg_train_loss)
    eval_losses.append(avg_eval_loss)
    wandb.log({"avg_train_loss": avg_train_loss})
    wandb.log({"eval_loss": avg_eval_loss})

    log_wandb_model(params, "autoencoder_{}_last".format(config['data_seq_len'], config['learning_rate']), 'model')

# Set a threshold to filter out outliers
thr = 100000

filtered_losses = [loss if loss <= thr else None for loss in losses]
filtered_eval_losses = [loss if loss <= thr else None for loss in eval_losses]

# # Save the ys_train image
# plt.imsave('ys_train_image.png', ys_train[0].T)

# # Close all figures to avoid memory leaks
# plt.close('all')