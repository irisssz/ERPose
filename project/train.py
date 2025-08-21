import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import sys
sys.path.append("..")
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.linemod.dataset_1 import PoseDataset_1 as PoseDataset_linemod_1
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
from itertools import cycle



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'linemod', help='select dataset')
parser.add_argument('--dataset_root', type=str, default = '../datasets/linemod/Linemod_preprocessed', help='dataset root dir')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=35, help='max number of epochs to train')

parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()



def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'linemod':
        opt.num_objects = 13 # number of object classes in the dataset
        opt.num_points = 500 # number of points on the input pointcloud
        opt.outf = '../trained_models/linemod'# folder to save trained models
        opt.log_dir = '../experiments/logs/linemod' # folder to save logs
        opt.repeat_epoch = 10 # number of repeat times for one epoch training
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()


    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
        opt.refine_start = True
        opt.decay_start = True
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        print("#########  Pre-trained model from {} is loaded   #########".format(opt.resume_posenet))
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    if opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
        test_dataset_1 = PoseDataset_linemod_1('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    testdataloader_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=1, shuffle=True, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf


    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0

        estimator.train()
        optimizer.zero_grad()
        
        # set the input dataset
        current_iter = cycle(dataloader)
        memory_iter = cycle(testdataloader_1)
        
        # set the switch frequency
        switch_frequency = 2 

        for rep in range(opt.repeat_epoch):
            for i in range(len(dataloader)):
                if i % switch_frequency == 0:
                    # training on memory buffer
                    memory_data = next(memory_iter)
                    mem_points, mem_choose, mem_img, mem_target, mem_model_points, mem_idx = memory_data
                    mem_points, mem_choose, mem_img, mem_target, mem_model_points, mem_idx = \
                                                                     Variable(mem_points).cuda(), \
                                                                     Variable(mem_choose).cuda(), \
                                                                     Variable(mem_img).cuda(), \
                                                                     Variable(mem_target).cuda(), \
                                                                     Variable(mem_model_points).cuda(), \
                                                                     Variable(mem_idx).cuda()

                    mem_pred_r, mem_pred_t, mem_pred_c, mem_emb = estimator(mem_img, mem_points, mem_choose, mem_idx)
                    loss_mem, dis_mem, new_points_mem, new_target_mem = criterion(mem_pred_r, mem_pred_t, mem_pred_c, mem_target, mem_model_points, mem_idx, mem_points, opt.w, opt.refine_start)
                    if i == 0:
                        loss_now = loss_mem
                        dis_now = dis_mem
                        
                else:
                    # training on current object
                    current_data = next(current_iter)
                    points, choose, img, target, model_points, idx = current_data
                    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                     Variable(choose).cuda(), \
                                                                     Variable(img).cuda(), \
                                                                     Variable(target).cuda(), \
                                                                     Variable(model_points).cuda(), \
                                                                     Variable(idx).cuda()

                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                    loss_now, dis_now, new_points_now, new_target_now = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)                    
                
                
                alpha = 1.0  # weight of current data loss
                beta = 1.0   # weight of memory buffer losses

                # compute tatal loss
                final_loss = alpha * loss_now + beta * loss_mem
                final_loss.backward(retain_graph=True)

                train_dis_avg += dis_now.item()
                train_count += 1
                
                print("train_count: {0}, loss_all: {1}, loss_ADD: {2}, loss_mem: {3}".format(train_count, final_loss, final_loss.item(), loss_mem.item()))

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            loss_test, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))

            test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            torch.save(estimator.state_dict(), '{0}/ERPose_pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        print("test finish")

if __name__ == '__main__':
    main()
