import os
import argparse
from collections import defaultdict
import time
import torchvision.transforms as transforms
import numpy as np
import torch
from torch import is_tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm
import sys
import wandb

from arguments import train_parser
from models import SwinIR
from utils.loss import get_loss
from utils.helper_functions import to_cuda, new_log, store_images
from dataloader.MagicBathy_new import MagicBathyNetDataLoader
from dataloader import MagicBathyNet


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_wandb = self.args.wandb
        self.dataloaders = self.get_dataloaders(args)
        self.model = SwinIR(upscale=args.upscale,
                            img_size=(64, 64),
                            in_chans=4,
                            patch_size=1,
                            window_size=args.window_size,
                            img_range=args.img_range,
                            depths=[8,8,8,8],
                            embed_dim=96,
                            num_heads=[8, 8, 8, 8],
                            upsampler='pixelshuffle').cuda()
        #self.img_store = args.img_store
        if self.use_wandb:
            self.experiment_folder = new_log(os.path.join(args.save_dir, args.dataset),
                                                                              args)[0]

            wandb.init(project=args.wandb_project, dir=self.experiment_folder)
            wandb.config.update(self.args)
            self.writer = None
        self.image_folder = os.path.join(os.getcwd(), 'save_images')
        self.experiment_name = args.experiment_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

    def __del__(self):
        if not self.use_wandb:
            self.writer.close()

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    self.scheduler.step()
                    if self.use_wandb:
                        wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)
                    else:
                        self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1




    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()
        with tqdm(self.dataloaders.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                self.optimizer.zero_grad()
                sample = to_cuda(sample)

                output = self.model(sample['source'])

                store_images(self.image_folder, self.experiment_name, output, sample["y"], self.epoch)
                loss = get_loss(output, sample)

                if torch.isnan(loss):
                    print('loss is nan')
                    continue
                    # raise Exception("detected NaN loss..")


                self.train_stats["loss"] += loss.detach().cpu().item()
                if self.epoch > 0 or not self.args.skip_first:
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                    self.optimizer.step()

                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders.datasets['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['loss'],
                                        validation_loss=self.val_stats['loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    if self.use_wandb:
                        wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    else:
                        for key in self.train_stats:
                            self.writer.add_scalar('train/' + key, self.train_stats[key], self.iter)

                    # reset metrics
                    self.train_stats = defaultdict(float)


    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders.dataloaders['val'], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample['source'])

                loss = get_loss(output, sample)
                self.val_stats["loss"] += loss.detach().cpu().item()


            self.val_stats = {k: v / len(self.dataloaders.datasets['val']) for k, v in self.val_stats.items()}

            if self.use_wandb:
                wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            else:
                for key in self.val_stats:
                    self.writer.add_scalar('val/' + key, self.val_stats[key], self.epoch)

            if self.val_stats['loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    def save_model(self, prefix=''):

        torch.save({
            'model': self.model.state_dict(),
            'epoch': self.epoch + 1,
            'iter': self.iter
        }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))




    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])

        if not args.no_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


    def get_dataloaders(self, args):
        if args.dataset == 'resized':
            norm_params = {
                "s2_an": np.load("./datasets/resized/agia_napa/norm_param_s2_an.npy"),
                "aerial_an": np.load("./datasets/resized/agia_napa/norm_param_aerial_an.npy"),
                #"s2_pl": np.load("./datasets/resized/puck_lagoon/norm_param_s2_pl.npy"),
                #"aerial_pl": np.load("./datasets/resized/puck_lagoon/norm_param_aerial_pl.npy"),
            }

            data_module = MagicBathyNetDataLoader(
                root_dir=os.path.join(os.getcwd(),"datasets", "resized"),
                locations=["agia_napa"],
                norm_params=norm_params,
                bathymetry=True,
                batch_size=4,
                num_workers=4,
                test_size=0.15,
                val_size=0.15,
                num_patches_per_image=40  # sample 10 patches per real image
            )

            train_loader = data_module.dataloaders["train"]
            val_loader = data_module.dataloaders["val"]
            test_loader = data_module.dataloaders["test"]

            return data_module

        if args.dataset == 'resized':
            path_to_images = os.path.join('.', 'datasets', 'resized', 'agia_napa', 'img', 's2')
            path_to_images = [os.path.join(path_to_images, x) for x in os.listdir(path_to_images)]
            path_to_labels = os.path.join('.', 'datasets', 'resized', 'agia_napa', 'img','aerial')

            return MagicBathyNet.MagicBathyNetDataLoader(
                os.path.join('.', 'datasets', 'resized'), batch_size=args.batch_size,
                num_workers=args.num_workers, locations=['agia_napa'], bathymetry=True)



if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    torch.cuda.empty_cache()  # Clear unused memory in PyTorch's cache
    args = train_parser.parse_args()
    print(train_parser.format_values())

    if args.wandb:
        import wandb

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))