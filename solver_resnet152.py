import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from networks_resnet152 import WGAN_ResNet152, WGAN_ResNet152_generator
from metric import compute_measure


class Solver(object):
    def __init__(self, mode, load_mode, data_loader, device, norm_range_min, norm_range_max, trunc_min, 
                 trunc_max, save_path, multi_gpu, num_epochs, print_iters, decay_iters, save_iters, 
                 model_name, result_fig, n_d_train, patch_n, batch_size, patch_size, lr, lambda_):

        self.mode = mode
        self.load_mode = load_mode
        self.data_loader = data_loader

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max
        self.trunc_min = trunc_min
        self.trunc_max = trunc_max

        self.save_path = save_path
        self.multi_gpu = multi_gpu

        self.num_epochs = num_epochs
        self.print_iters = print_iters
        self.decay_iters = decay_iters
        self.save_iters = save_iters
        self.model_name = model_name
        self.result_fig = result_fig

        self.n_d_train = n_d_train

        self.patch_n = patch_n
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.lr = lr
        self.lambda_ = lambda_

        self.WGAN_ResNet152 = WGAN_ResNet152(input_size=patch_size if patch_n else 512)

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.WGAN_ResNet152 = nn.DataParallel(self.WGAN_ResNet152)

        self.WGAN_ResNet152.to(self.device)

        self.criterion_perceptual = nn.L1Loss()
        self.optimizer_g = optim.Adam(self.WGAN_ResNet152.generator.parameters(), self.lr)
        self.optimizer_d = optim.Adam(self.WGAN_ResNet152.discriminator.parameters(), self.lr)


    def save_model(self, iter_, loss_=None):
        f = os.path.join(self.save_path, 'WGAN_ResNet152_{}iter.ckpt'.format(iter_))
        torch.save(self.WGAN_ResNet152.state_dict(), f)
        if loss_:
            f_loss = os.path.join(self.save_path, 'WGAN_ResNet152_loss_{}iter.npy'.format(iter_))
            np.save(f_loss, np.array(loss_))


    def load_model(self, model_name):
        f = 'models/{}.ckpt'.format(model_name)
        generator_w = {k[10:]:torch.load(f)[k] for k in  list(torch.load(f).keys()) if 'generator' in k}
        
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in generator_w:
                n = k[7:]
                state_d[n] = v
            self.WGAN_ResNet152_G.load_state_dict(state_d)
        else:
            self.WGAN_ResNet152_G.load_state_dict(generator_w)


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        save_fig_folder = os.path.join('results', self.model_name)
        if not os.path.exists(save_fig_folder):
            os.mkdir(save_fig_folder)
        f.savefig(os.path.join('results', self.model_name, 'result_{}.png'.format(fig_name)))
        plt.close()


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()

        for epoch in range(1, self.num_epochs):
            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1
                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                # patch training
                if self.patch_size:
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                # discriminator
                self.optimizer_d.zero_grad()
                self.WGAN_ResNet152.discriminator.zero_grad()
                for _ in range(self.n_d_train):
                    d_loss, gp_loss = self.WGAN_ResNet152.d_loss(x, y, gp=True, return_gp=True)
                    d_loss.backward()
                    self.optimizer_d.step()

                # generator, perceptual loss
                self.optimizer_g.zero_grad()
                self.WGAN_ResNet152.generator.zero_grad()
                g_loss, p_loss = self.WGAN_ResNet152.g_loss(x, y, perceptual=True, return_p=True)
                g_loss.backward()
                self.optimizer_g.step()

                train_losses.append([g_loss.item()-p_loss.item(), p_loss.item(),
                                     d_loss.item()-gp_loss.item(), gp_loss.item()])

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}], TIME [{:.1f}s]\nG_LOSS: {:.8f}, D_LOSS: {:.8f}".format(total_iters, epoch, self.num_epochs, iter_ + 1, len(self.data_loader), time.time() - start_time, g_loss.item(), d_loss.item()))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters, train_losses)
        self.save_model(total_iters, train_losses)

    def test(self):
        del self.WGAN_ResNet152
        # load
        self.WGAN_ResNet152_G = WGAN_ResNet152_generator().to(self.device)
        self.load_model(self.model_name)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.WGAN_ResNet152_G(x)

                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), ori_ssim_avg/len(self.data_loader), ori_rmse_avg/len(self.data_loader)))
            print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), pred_ssim_avg/len(self.data_loader), pred_rmse_avg/len(self.data_loader)))