import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
import torch.optim as optim

import pandas as pd

import numpy as np; np.set_printoptions(precision=4)
import shutil, argparse, time
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from src import config
from src.data import collate_remove_none, collate_stack_together, worker_init_fn
from src.training import Trainer
from src.model import Encode2Points
from src.utils import load_config, initialize_logger, \
AverageMeter, load_model_manual


def main():

    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--gpu', type=str, help='which gpu to use.')

    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:"+args.gpu if use_cuda else "cpu")
    input_type = cfg['data']['input_type']
    batch_size = cfg['train']['batch_size']
    model_selection_metric = cfg['train']['model_selection_metric']


    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    # boiler-plate
    if cfg['train']['timestamp']:
        cfg['train']['out_dir'] += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = initialize_logger(cfg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    shutil.copyfile(args.config, os.path.join(cfg['train']['out_dir'], 'config.yaml'))

    logger.info("using GPU: " + torch.cuda.get_device_name(0))

    # TensorboardX writer
    tblogdir = os.path.join(cfg['train']['out_dir'], "tensorboard_log")
    if not os.path.exists(tblogdir):
        os.makedirs(tblogdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tblogdir)

    
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)

    
    collate_fn = collate_remove_none

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=cfg['train']['n_workers'], shuffle=True,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['train']['n_workers_val'], shuffle=False,
    collate_fn=collate_remove_none,
    worker_init_fn=worker_init_fn)
    
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(Encode2Points(cfg)).to(device)
    # else:
    model = Encode2Points(cfg).to(device)

    n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of parameters: %d'% n_parameter)
    # load model
    try:
        # load model
        state_dict = torch.load(os.path.join(cfg['train']['dir_model'],'model_best.pt'))
        load_model_manual(state_dict['state_dict'], model)

        out = "Load model from iteration %d" % state_dict.get('it', 0)
        logger.info(out)
        # load point cloud
    except:
        print("Model could not be loaded from {}".format(os.path.join(os.path.join(cfg['train']['dir_model'], 'model.pt'))))
        state_dict = dict()
    

    metric_val_best = state_dict.get(
    'iou_val_best', np.inf)
    logger.info('Current best IoU (%s): %.8f'
      % (model_selection_metric, metric_val_best))
    metric_val_best = state_dict.get(
    'loss_val_best', np.inf)
    logger.info('Current loss (%s): %.8f'
      % (model_selection_metric, metric_val_best))

    LR = float(cfg['train']['lr'])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch = state_dict.get('epoch', -1)
    it = state_dict.get('it', -1)

    trainer = Trainer(cfg, optimizer, device=device)
    runtime = {}
    runtime['all'] = AverageMeter()



    # create the results df
    cols = ['iteration', 'epoch','train_loss', 'test_loss', 'test_best_loss', 'test_iou', 'test_best_iou']
    results_df = pd.DataFrame(columns=cols)
    os.makedirs(os.path.join(cfg['train']['out_dir'],"metrics"),exist_ok=True)
    best_iou = 0.0
    current_loss = []
    # training loop
    for epoch in range(start_epoch+1, cfg['train']['total_epochs']+1):
        for batch in train_loader:
            it += 1

            start = time.time()
            loss, loss_each = trainer.train_step(batch, model)
            current_loss.append(loss)
            # measure elapsed time
            end = time.time()
            runtime['all'].update(end - start)



            if it % cfg['train']['print_every'] == 0:
                loss = np.array(current_loss).mean()
                log_text = ('[Epoch %02d] it=%d, loss=%.4f') %(epoch, it, loss)
                writer.add_scalar('train/loss', loss, it)
                if loss_each is not None:
                    for k, l in loss_each.items():
                        if l.item() != 0.:
                            log_text += (' loss_%s=%.4f') % (k, l.item())
                        writer.add_scalar('train/%s' % k, l, it)
                
                log_text += (' time=%.3f / %.2f') % (runtime['all'].val, runtime['all'].sum)
                logger.info(log_text)
                current_loss = []

            if (it>0)& (it % cfg['train']['visualize_every'] == 0):
                for i, batch_val in enumerate(val_loader):
                    trainer.save(model, batch_val, it, i)
                    if i >= 4:
                        break
                logger.info('Saved mesh and pointcloud')

            # run validation
            if it > 0 and (it % cfg['train']['validate_every']) == 0:

                row = dict.fromkeys(list(results_df.columns))

                eval_dict = trainer.evaluate(val_loader, model)
                metric_val = eval_dict[model_selection_metric]
                logger.info('Validation metric (%s): %.4f'
                    % (model_selection_metric, metric_val))
                
                for k, v in eval_dict.items():
                    writer.add_scalar('val/%s' % k, v, it)

                if  -(metric_val - metric_val_best) >= 0:
                    metric_val_best = metric_val
                    logger.info('New best model (loss %.4f)' % metric_val_best)
                    state = {'epoch': epoch,
                            'it': it,
                            'loss_val_best': metric_val_best}
                    # state['state_dict'] = model.state_dict()
                    # torch.save(state, os.path.join(cfg['train']['dir_model'], 'model_best.pt'))

                mean_iou = []
                for i, batch_val in enumerate(tqdm(val_loader,ncols=50)):
                    mean_iou.append(trainer.generate_and_evaluate(model, batch_val))

                mean_iou = np.array(mean_iou).mean()
                if(mean_iou > best_iou):
                    logger.info('New best model (IoU %.4f)' % mean_iou)
                    best_iou = mean_iou
                    best_epoch = epoch
                    state = {'epoch': epoch,
                             'it': it,
                             'loss_val_best': metric_val_best,
                             'iou_val_best': best_iou}
                    state['state_dict'] = model.state_dict()
                    torch.save(state, os.path.join(os.path.join(cfg['train']['dir_model'],'model_best.pt')))

                row["iteration"] = it
                row["epoch"] = epoch
                row["train_loss"] = loss
                row["test_loss"] = metric_val
                row["test_best_loss"] = metric_val_best
                row["test_iou"] = mean_iou
                row["test_best_iou"] = best_iou
                ### write metrics to file
                results_df = results_df.append(row, ignore_index=True)
                results_file = os.path.join(os.path.join(cfg['train']['out_dir'],"metrics","results.csv"))
                results_df.to_csv(results_file, index=False)

                # print(row)
                logger.info(row)

            # save checkpoint
            if (epoch > 0) & (it % cfg['train']['checkpoint_every'] == 0):
                state = {'epoch': epoch,
                         'it': it,
                         'loss_val_best': metric_val_best,
                         'iou_val_best': best_iou}
                state['state_dict'] = model.state_dict()
                torch.save(state, os.path.join(cfg['train']['dir_model'], 'model.pt'))

                if (it % cfg['train']['backup_every'] == 0):
                    torch.save(state, os.path.join(cfg['train']['dir_model'], '%04d' % it + '.pt'))
                    logger.info("Backup model at iteration %d" % it)

                    results_file = os.path.join(os.path.join(cfg['train']['out_dir'], "metrics", "results_" + str(it) + ".csv"))
                    results_df.to_csv(results_file, index=False)

                logger.info("Save new model at iteration %d" % it)

            done=time.time()

if __name__ == '__main__':
    main()