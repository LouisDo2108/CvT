from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle as pkl
import pprint
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter

import _init_paths
from config import config
from config import update_config
from core.function import test
from core.loss import build_criterion
from dataset import build_dataloader
from dataset import RealLabelsImagenet
from models import build_model
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master
from utils.utils import strip_prefix_if_present

from pathlib import Path
import numpy as np

def _meter_reduce(meter):
    rank = comm.local_rank
    meter_sum = torch.FloatTensor([meter.sum]).cuda(rank)
    meter_count = torch.FloatTensor([meter.count]).cuda(rank)
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    
def write_output_to_txt(p, outputs, y, one_five='five',
                        output_path="/root/data/ltnghia/projects/visual_communication/htluc/CvT/OUTPUT/imagenet/cvt-13-224x224_esr/"):
    output_name = ""
    y = y.cpu().numpy()
    if one_five == 'five':
        _, outputs = torch.topk(outputs, 5, dim=1)
        outputs = outputs.cpu().numpy()
        output_name = "top5_output.txt"
    elif one_five == 'one':
        outputs = outputs.argmax(dim=1).cpu().numpy()
        output_name = "top1_output.txt"
    # print("Output shape after", outputs.shape)
    # print("y shape after", y.shape)
    
    result = np.array(list(zip(p, outputs, y)))
    with open(os.path.join(output_path, output_name), 'a') as f:
        for img_path, pred, gt in result:
            f.write("{}, ({}), {}\n".format(img_path, pred, gt))
    return 

@torch.no_grad()
def test(config, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False, real_labels=None,
         valid_labels=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    # for i, (x, y) in enumerate(val_loader):
    for i, (p, x, y) in enumerate(val_loader):

        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        outputs = model(x)
        if valid_labels:
            outputs = outputs[:, valid_labels]
        
        write_output_to_txt(p, outputs, y, output_path=output_dir)
        
        loss = criterion(outputs, y)

        if real_labels and not distributed:
            real_labels.add_result(outputs)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        prec1, prec5 = accuracy(outputs, y, (1, 5))
        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # break

    logging.info('=> synchronize...')
    comm.synchronize()
    top1_acc, top5_acc, loss_avg = map(
        _meter_reduce if distributed else lambda x: x.avg,
        [top1, top5, losses]
    )

    if real_labels and not distributed:
        real_top1 = real_labels.get_accuracy(k=1)
        real_top5 = real_labels.get_accuracy(k=5)
        msg = '=> TEST using Reassessed labels:\t' \
            'Error@1 {error1:.3f}%\t' \
            'Error@5 {error5:.3f}%\t' \
            'Accuracy@1 {top1:.3f}%\t' \
            'Accuracy@5 {top5:.3f}%\t'.format(
                top1=real_top1,
                top5=real_top5,
                error1=100-real_top1,
                error5=100-real_top5
            )
        logging.info(msg)

    if comm.is_main_process():
        msg = '=> TEST:\t' \
            'Loss {loss_avg:.4f}\t' \
            'Error@1 {error1:.3f}%\t' \
            'Error@5 {error5:.3f}%\t' \
            'Accuracy@1 {top1:.3f}%\t' \
            'Accuracy@5 {top5:.3f}%\t'.format(
                loss_avg=loss_avg, top1=top1_acc,
                top5=top5_acc, error1=100-top1_acc,
                error5=100-top5_acc
            )
        logging.info(msg)

    # if writer_dict and comm.is_main_process():
    #     writer = writer_dict['writer']
    #     global_steps = writer_dict['valid_global_steps']
    #     writer.add_scalar('valid_loss', loss_avg, global_steps)
    #     writer.add_scalar('valid_top1', top1_acc, global_steps)
    #     writer_dict['valid_global_steps'] = global_steps + 1

    # logging.info('=> switch to train mode')
    # model.train()

    return top1_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    args.distributed = False
    args.num_gpus = 1
    # init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'test')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))

    model = build_model(config)
    model.to(torch.device('cuda'))

    model_file = config.TEST.MODEL_FILE if config.TEST.MODEL_FILE \
        else os.path.join(final_output_dir, 'model_best.pth')
    # model_file = "/root/data/ltnghia/projects/visual_communication/htluc/CvT/OUTPUT/imagenet/cvt-13-224x224_esr/model_best.pth"
    logging.info('=> load model file: {}'.format(model_file))
    ext = model_file.split('.')[-1]
    if ext == 'pth':
        state_dict = torch.load(model_file, map_location="cpu")
    else:
        raise ValueError("Unknown model file")

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # summary_model_on_master(model, config, final_output_dir, False)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.local_rank], output_device=args.local_rank
    #     )

    # define loss function (criterion) and optimizer
    criterion = build_criterion(config, train=False)
    criterion.cuda()

    valid_loader = build_dataloader(config, False, args.distributed)
    real_labels = None
    if (
        config.DATASET.DATASET == 'imagenet'
        and config.DATASET.DATA_FORMAT == 'tsv'
        and config.TEST.REAL_LABELS
    ):
        filenames = valid_loader.dataset.get_filenames()
        real_json = os.path.join(config.DATASET.ROOT, 'real.json')
        logging.info('=> loading real labels...')
        real_labels = RealLabelsImagenet(filenames, real_json)

    valid_labels = None
    if config.TEST.VALID_LABELS:
        with open(config.TEST.VALID_LABELS, 'r') as f:
            valid_labels = {
                int(line.rstrip()) for line in f
            }
            valid_labels = [
                i in valid_labels for i in range(config.MODEL.NUM_CLASSES)
            ]

    logging.info('=> start testing')
    start = time.time()
    test(config, valid_loader, model, criterion,
         final_output_dir, tb_log_dir, writer_dict,
         args.distributed, real_labels=real_labels,
         valid_labels=valid_labels)
    logging.info('=> test duration time: {:.2f}s'.format(time.time()-start))

    writer_dict['writer'].close()
    logging.info('=> finish testing')


if __name__ == '__main__':
    main()
