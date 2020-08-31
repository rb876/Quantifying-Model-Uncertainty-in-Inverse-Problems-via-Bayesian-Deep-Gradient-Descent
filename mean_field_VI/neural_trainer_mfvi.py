import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import layer_utils as utils

class NeuralTrainer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, grad, num_samples=1):
        raise NotImplementedError

    def optimise(self, train_loader, epochs=100, batch_size=64, initial_lr=1e-3, weight_decay=1e-3):

        weights_phi = [v for k, v in self.named_parameters() if k.startswith('det_')]
        weights = [v for k, v in self.named_parameters() if (not k.startswith('det_'))]
        optimizer = optim.Adam([{'params': weights_phi, 'weight_decay': weight_decay},
                                {'params': weights}], lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-4)
        for epoch in range(epochs):
            scheduler.step()
            losses, kls, rmses, psnrs = [], [], [], []
            for (data, grad, target) in train_loader:
                self.train()
                optimizer.zero_grad()
                data, grad, target = utils.to_gpu(data, grad, target)
                x_pred = self.forward(data, grad, num_samples=1)
                step_loss, kl = self._compute_loss(target.squeeze(), x_pred.squeeze(), len(target), len(train_loader.dataset))
                step_loss.backward()
                optimizer.step()

                rmse, psnr = self._evaluate_performance(target, x_pred)
                losses.append(step_loss.cpu().item())
                kls.append(kl.cpu().item())
                rmses.append(rmse)
                psnrs.append(psnr)

            print('epoch: {}, loss: {:.4f}, kl: {:.4f}, rmse: {:.8f}, psnr: {:.8f}'\
                .format(epoch, np.mean(losses), np.mean(kls), np.mean(rmses), np.mean(psnrs)), flush=True)

    def test(self, val_loader):
        print('Testing...', flush=True)
        test_bsz = 1024
        losses, performances = self._evaluate(val_loader, test_bsz)
        return np.hstack(losses), np.hstack(performances)

    def _compute_loss(self, x, x_pred, batch_size, data_size):
        raise NotImplementedError

    def _compute_log_likelihood(self, x, x_pred, var):
        return torch.sum(utils.gaussian_log_density(inputs=utils._squeeze(x), mean=utils._squeeze(x_pred), variance=var), dim=0)

    def _evaluate_performance(self, x, x_pred):
        from math import log10
        return (torch.sqrt(torch.mean((x - x_pred) ** 2)).cpu().item(), 10
                * log10(1 / torch.mean((x - x_pred) ** 2)))

    def _evaluate(self, val_loader, batch_size):
        losses, performances = [], []
        self.eval()
        with torch.no_grad():
            for (data, grad, target) in val_loader:
                data, grad, target = utils.to_gpu(data, grad, target)
                x_pred_samples = self.forward(data, grad, num_samples=1)
                loss = self._compute_log_likelihood(target, x_pred_samples, self.log_noise.exp()**2)
                avg_loss = loss / len(target)
                performance = self._evaluate_performance(target, x_pred_samples)[1]
                losses.append(avg_loss.cpu().item())
                performances.append(performance)

        return losses, performances
