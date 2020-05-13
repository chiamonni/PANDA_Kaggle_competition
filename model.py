from network import DenseNet201
from PANDA_metric import PANDA_metric, PANDA_loss
from PANDA_functions import QWKMetric, QWKLoss

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from typing import Dict, Union

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


class Model:
    def __init__(self,
                 #net_type: str,
                 #net_hyperparams: Dict[str, Union[str, int, float]],
                 optimizer_type: str = 'adam',
                 #loss_type: str = 'kappa_loss',  # SmoothL1Loss
                 lr: float = .01,
                 ):
        #self.net_type = net_type
        #self.net_hyperparams = net_hyperparams
        self.optimizer = optimizer_type
        #self.loss = loss_type
        self.lr = lr

        self.metric, self.loss, self.optimizer, self.net = self.__build_model()

    def __build_model(self) -> (nn.Module, nn.Module, torch.optim, nn.Module):
        # Mandatory parameters to be used.
        # dropout_prob = self.net_hyperparams['dropout_prob']
        # num_init_features = self.net_hyperparams['num_init_features']

        network: nn.Module = DenseNet201()

        # Define metric, loss, optimizer
        # metric_fn = PANDA_metric()  # Define metric function
        #loss_fn = PANDA_loss()
        metric_fn = QWKMetric()
        loss_fn = QWKLoss()


        if self.optimizer == 'adam':
            # Define the optimizer. It wants to know which parameters are being optimized.
            optimizer_fn = torch.optim.Adam(params=network.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.optimizer == 'SGD':
            optimizer_fn = torch.optim.SGD(params=network.parameters(), lr=self.lr, momentum=0.3, weight_decay=1e-7)
        else:
            raise ValueError('Bad optimizer type. Please choose adam or ...')

        return metric_fn, loss_fn, optimizer_fn, network

    '''
    def __save(self, run_path, metric, epoch):
        state = {
            'state_dict': self.net.state_dict(),
            'optim_dict': self.optimizer.state_dict()
        }
        filepath = os.path.join(run_path, 'checkpoint_' + str(metric) + '_ep' + str(epoch) + '.pt')
        torch.save(state, filepath)
    '''

    def fit(self, epochs, train_loader, val_loader):
        #early_stopping = EarlyStopping(patience=patience, verbose=False)

        #cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_loader), 1e-8)
        #on_plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=4)

        start_epoch = torch.cuda.Event(enable_timing=True)
        start_whole = torch.cuda.Event(enable_timing=True)
        end_whole = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)

        start_whole.record()

        for epoch in range(epochs):
            start_epoch.record()

            train_loss, train_metric = self.net.train_batch(self.net, train_loader, self.loss, self.metric, self.optimizer, DEVICE)
            val_loss, val_metric = self.net.val_batch(self.net, val_loader, self.loss, self.metric, DEVICE)

            end_epoch.record()
            torch.cuda.synchronize(DEVICE)
            # Calculate elapsed time
            elapsed_seconds = start_epoch.elapsed_time(
                        end_epoch) / 1000
            elapsed_minutes = elapsed_seconds // 60
            elapsed_seconds = round(elapsed_seconds % 60)
            print(
                "\nEpoch: {}\ttrain metric: {:.4f} loss: {:.4f}\t\tval metric: {:.4f} loss: {:.4}\ttime: {:.0f}m{:.0f}s".format(
                    epoch+1,
                    train_metric,
                    train_loss,
                    val_metric,
                    val_loss,
                    elapsed_minutes,
                    elapsed_seconds
                ))
            # Update early stopping. This is really useful to stop training in time.
            # The if statement is not slowing down training since each epoch last very long.
            #epoch_val_metric = val_metric.item()
            #poch_train_metric = train_metric.item()
          #  early_stopping(epoch_train_metric, epoch_val_metric, self.net)
           # if early_stopping.save_checkpoint and run_path:
            #    self.__save(run_path, epoch_val_metric, epoch)
            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    break

            #on_plateau_scheduler.step(val_metric)

        end_whole.record()
        torch.cuda.synchronize(DEVICE)
        print("Elapsed time: {:.4f}m".format(start_whole.elapsed_time(end_whole) / 60000))

        # Return the best metric that we register for early stopping.
        #return early_stopping.val_metric_min

    '''
    def submit(self, test_loader, run_path):
        IDs, outputs = self.net.predict_batch(self.net, test_loader, DEVICE)
        submission = pd.DataFrame(columns=['Id', 'Predicted'])
        sub_names = [
            '_age',
            '_domain1_var1',
            '_domain1_var2',
            '_domain2_var1',
            '_domain2_var2'
        ]

        for ID, output in tqdm(zip(IDs, outputs), desc='Writing predictions on submission.csv file...', total=len(outputs)):
            sub_names_part = [str(int(ID)) + sn for sn in sub_names]
            for name, out in zip(sub_names_part, output):
                submission.loc[len(submission['Id'])] = [name, out]
        submission.to_csv(os.path.join(run_path, 'submission.csv'), index=False)
    '''
