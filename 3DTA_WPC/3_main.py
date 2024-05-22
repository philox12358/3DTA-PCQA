import os
import argparse
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_load import WPC_SD
from model_3DTA import Pct_3DTA
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream

from tqdm import tqdm
from scipy import stats
from datetime import datetime
import time



def copy_code(results_dir):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    shutil.copy('1_pc_to_patch.py', f'{results_dir}/1_pc_to_patch.py')
    shutil.copy('2_patch_list_create.py', f'{results_dir}/2_patch_list_create.py')
    shutil.copy('3_main.py', f'{results_dir}/3_main.py')
    shutil.copy('data_load.py', f'{results_dir}/data_load.py')
    shutil.copy('model_3DTA.py', f'{results_dir}/model_3DTA.py')
    shutil.copy('util.py', f'{results_dir}/util.py')



def train(args):

    train_data = WPC_SD(args, pattern='train')
    train_loader = DataLoader(train_data, num_workers=4,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_data = WPC_SD(args, pattern='test')
    test_loader = DataLoader(test_data, num_workers=4,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = Pct_3DTA(args).to(device).float()
    # print(str(model))
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD...")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                    momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    mse_criterion = nn.MSELoss()
    best_test_plcc = -10000.0
    best_test_record = 'no info'
    model_path = f'./_model.pth'
    if args.pre_train and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('\033[1;35mUSE pretrained model... \033[0m')
    else:
        print(f'没有/不使用预训练的模型...')
        
    begin_time = time.time()
    for epoch in range(args.epochs):
        #?###################
        #? Train
        #?###################
        train_plcc_loss = 0.0
        train_mae_loss = 0.0
        train_total_loss = 0.0
        train_count = 0.0
        model.train()    # training turn on
        train_ply_num = int(len(train_data)/args.patch_num)
        filenum_mos_true = [0]*train_ply_num
        filenum_mos_pred = [0]*train_ply_num

        for id, (data, mos, filenum) in tqdm(enumerate(train_loader, 0), 
                total=len(train_loader), smoothing=0.9, desc =f'train epoch: {epoch}', colour = 'blue'):
            data, mos = data.to(device), mos.to(torch.float64).to(device).squeeze()
            data = data.permute(0, 2, 1)
            data = data.type(torch.FloatTensor)
            batch_size = data.size()[0]
            optimizer.zero_grad()
            pre_mos = model(data)              #?@@@@@@@@@@@@@@@@@@@@@@@  train forward
            pre_mos = pre_mos.to(torch.float64).view(batch_size)
            pre_mos_cpu = (pre_mos).detach().cpu().numpy()    
            true_mos_cpu = (mos).cpu().numpy()

            loss = mse_criterion(pre_mos, mos)

            loss.backward()
            optimizer.step()
            train_count += batch_size
            train_total_loss += loss.item() * batch_size
            
            for i in range(batch_size):
                filenum_mos_pred[int(filenum[i])] += pre_mos_cpu[i]
                filenum_mos_true[int(filenum[i])] = true_mos_cpu[i]
        scheduler.step()        
        filenum_mos_true = torch.tensor(filenum_mos_true)               # list2Tensor
        filenum_mos_pred = torch.tensor(filenum_mos_pred)
        filenum_mos_pred = filenum_mos_pred / args.patch_num
        ply_train_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]   # PLCC
        ply_train_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]  # SRCC
        ply_train_rmse = torch.sqrt(((filenum_mos_true - filenum_mos_pred)**2).mean())  # RMSE

        record = f'Train {epoch:3d},  loss:{train_total_loss*1.0/train_count:.4f}, PLCC:{ply_train_PLCC:.4f}, SRCC:{ply_train_SRCC:.4f}, rmse:{ply_train_rmse:.4f}'
        print(record)

        time_now = f'{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}'
        with open(args.results_dir + '/train_log.txt', 'a+') as txt:
            txt.write(f'\n{time_now}    {record}')




        #*###################
        #* Test
        #*###################
        test_ply_num = int(len(test_data)/args.patch_num)
        test_plcc_loss = 0.0
        test_mae_loss = 0.0
        test_total_loss = 0.0
        test_count = 0.0
        model.eval()     # training turn off
        filenum_mos_true = [0]*test_ply_num
        filenum_mos_pred = [0]*test_ply_num

        for id, (data, mos, filenum) in tqdm(enumerate(test_loader, 0), 
                total=len(test_loader), smoothing=0.9, desc =f'test  epoch：{epoch}', colour = 'green'):
            data, mos = data.to(device), mos.to(device).squeeze()
            data = data.permute(0, 2, 1)
            data = data.type(torch.FloatTensor)
            batch_size = data.size()[0]
            pre_mos = model(data)                  #*@@@@@@@@@@@@@@@@@@@@@@@  test forward
            pre_mos = pre_mos.to(torch.float64).view(batch_size)
            pre_mos_cpu = (pre_mos).detach().cpu().numpy()
            true_mos_cpu = (mos).cpu().numpy()

            loss = mse_criterion(pre_mos, mos)

            # preds = logits.max(dim=1)[1]            # for classfication
            test_count += batch_size
            test_total_loss += loss.item() * batch_size

            for i in range(batch_size):
                filenum_mos_pred[int(filenum[i])] += pre_mos_cpu[i]
                filenum_mos_true[int(filenum[i])] = true_mos_cpu[i]

        filenum_mos_true = torch.tensor(filenum_mos_true)           # list2Tensor
        filenum_mos_pred = torch.tensor(filenum_mos_pred)
        filenum_mos_pred = filenum_mos_pred / args.patch_num
        ply_test_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]    # PLCC
        ply_test_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]   # SRCC
        ply_test_rmse = torch.sqrt(((filenum_mos_true - filenum_mos_pred)**2).mean())   # RMSE

        record = f'Test  {epoch:3d},  loss:{test_total_loss*1.0/test_count:.4f}, PLCC:{ply_test_PLCC:.4f}, SRCC:{ply_test_SRCC:.4f}, rmse:{ply_test_rmse:.4f}'
        print(record)

        time_now = f'{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}'
        with open(args.results_dir + '/train_log.txt', 'a+') as txt:
            txt.write(f'\n{time_now}    {record}')

        if ply_test_PLCC > best_test_plcc:           # Find the best model and save it
            best_test_plcc = ply_test_PLCC
            best_test_record = record
            torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), f'{args.results_dir}/{model_path}')
            print(f'\033[1;35mTime now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
            print(f'{best_test_record}\033[0m')
            print(f'filenum_mos_true:{filenum_mos_true}')            
            print(f'filenum_mos_pred:{filenum_mos_pred} \033[0m')
            with open(args.results_dir + '/train_log.txt', 'a+') as txt:
                txt.write(f'  @@@ Best @@@  ')

        if epoch==100:                               # record score epoch=100
            cost_time = time.time()-begin_time
            print(f'\033[1;35mTime now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
            print(f'Time in 100 epoch:  {cost_time/60:.4f}  minute...')
            print(f'BEST_Record: {best_test_record}')
            print(f'best_test_plcc: {best_test_plcc}')
            print(f'filenum_mos_true:{filenum_mos_true}')            
            print(f'filenum_mos_pred:{filenum_mos_pred} \033[0m')
            




def test(args):
    print('start test...')
    test_data = WPC_SD(args, pattern='test')
    test_loader = DataLoader(test_data, num_workers=4,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = Pct_3DTA(args).to(device)
    model = nn.DataParallel(model)
    model_path = f'_model.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.eval()       #  training turn off

    test_ply_num = int(len(test_data)/args.patch_num)
    test_count = 0.0
    filenum_mos_true = [0]*test_ply_num
    filenum_mos_pred = [0]*test_ply_num
    show_all_mos = torch.zeros([test_ply_num, int(args.patch_num)])
    for id, (data, mos, filenum) in tqdm(enumerate(test_loader, 0), 
        total=len(test_loader), smoothing=0.9, desc =f'Just test', colour = 'green'):
        data, mos = data.to(device), mos.to(device).squeeze()
        data = data.permute(0, 2, 1)
        data = data.type(torch.FloatTensor)
        batch_size = data.size()[0]
        pre_mos = model(data)                  #*@@@@@@@@@@@@@@@@@@@@@@@  evaluate
        pre_mos = pre_mos.to(torch.float64).view(batch_size)
        pre_mos_cpu = (pre_mos).detach().cpu().numpy()
        true_mos_cpu = (mos).cpu().numpy()
        # preds = logits.max(dim=1)[1]
        test_count += batch_size
        for i in range(batch_size):
            filenum_mos_pred[int(filenum[i])] += pre_mos_cpu[i]
            filenum_mos_true[int(filenum[i])] = true_mos_cpu[i]

            for id, mos in enumerate(show_all_mos[int(filenum[i])]):
                if mos == 0:
                    show_all_mos[int(filenum[i])][id] = pre_mos_cpu[i]
                    break

    show_all_mos = show_all_mos.permute(1,0)


    filenum_mos_true = torch.tensor(filenum_mos_true)
    filenum_mos_pred = torch.tensor(filenum_mos_pred)
    filenum_mos_pred = filenum_mos_pred / args.patch_num
    ply_test_PLCC = stats.mstats.pearsonr(filenum_mos_true, filenum_mos_pred)[0]   # calculate corelation
    ply_test_SRCC = stats.mstats.spearmanr(filenum_mos_true, filenum_mos_pred)[0]
    ply_test_rmse = torch.sqrt(((filenum_mos_true - filenum_mos_pred)**2).mean())
    print(f'\033[1;35mTest (ply) {test_ply_num},    PLCC:{ply_test_PLCC:.4f},  SRCC:{ply_test_SRCC:.4f}', end='')
    print(f',  rmse:{ply_test_rmse:.4f}\n')

    print(f'Time now: {time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())}')
    print(f'filenum_mos_true:{filenum_mos_true}')
    print(f'filenum_mos_pred:{filenum_mos_pred}\033[0m')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Our 3DTA')

    parser.add_argument('--exp_name', type=str, default='3DTA_patch_mos', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--point_num', type=int, default=1024, help='num of points to use')

    parser.add_argument('--pre_train', type=bool,  default=False, help='evaluate the model?')
    parser.add_argument('--eval', type=bool,  default=False, help='evaluate the model?')
    
    parser.add_argument('--data_dir', type=str, default='../data/WPC', metavar='N', help='Where does dataset exist?')
    parser.add_argument('--patch_dir', type=str, default='patch_72_10000', help='Where does patches exist?')
    parser.add_argument('--patch_num', type=int, default=72, metavar='N', help='How many patchs each PC have?')

    args = parser.parse_args()
    print(args)


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        print(f'Using GPU :{torch.cuda.current_device()} from {torch.cuda.device_count()}devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')


    if args.eval:
        test(args)       # evaluate begin
    else:
        args.results_dir = './checkpoints/Train_MOS_' + datetime.now().strftime("%m-%d_%H-%M-%S")
        os.mkdir(args.results_dir)
        copy_code(args.results_dir)      # save related file
        train(args)      # train begin


