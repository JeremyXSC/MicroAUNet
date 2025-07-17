import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import math

from models.maunet import MAUNet
from models.malunet import MALUNet
from dataset.npy_datasets import NPY_datasets
from dataset.cvc_datasets import CVC_datasets
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, pos_weight=1.0, neg_weight=1.0, 
                 max_pos_samples=512, max_neg_samples=1024):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.max_pos_samples = max_pos_samples
        self.max_neg_samples = max_neg_samples
        
    def forward(self, student_pred, teacher_pred, targets):
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # [B, H, W]
        
        # 获取预测概率
        student_prob = F.softmax(student_pred / self.temperature, dim=1)
        teacher_prob = F.softmax(teacher_pred / self.temperature, dim=1)
        
        batch_size, num_classes, height, width = student_pred.shape
        
        pos_mask = (targets > 0.5)
        neg_mask = (targets <= 0.5)
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            pos_indices = torch.where(pos_mask[b])
            neg_indices = torch.where(neg_mask[b])
            
            if len(pos_indices[0]) < 2 or len(neg_indices[0]) == 0:
                continue
                
            num_pos = len(pos_indices[0])
            num_neg = len(neg_indices[0])
            
            # 正样本采样
            if num_pos > self.max_pos_samples:
                pos_sample_idx = torch.randperm(num_pos, device=student_pred.device)[:self.max_pos_samples]
                pos_indices = (pos_indices[0][pos_sample_idx], pos_indices[1][pos_sample_idx])
                num_pos = self.max_pos_samples
            
            # 负样本采样
            if num_neg > self.max_neg_samples:
                neg_sample_idx = torch.randperm(num_neg, device=student_pred.device)[:self.max_neg_samples]
                neg_indices = (neg_indices[0][neg_sample_idx], neg_indices[1][neg_sample_idx])
                num_neg = self.max_neg_samples
            
            # 提取特征向量
            pos_features = student_prob[b, :, pos_indices[0], pos_indices[1]].T  # [N_pos, C]
            neg_features = student_prob[b, :, neg_indices[0], neg_indices[1]].T  # [N_neg, C]
            
            # 归一化特征
            pos_features = F.normalize(pos_features, dim=1, p=2)
            neg_features = F.normalize(neg_features, dim=1, p=2)
            
            # 计算损失
            batch_loss = self._compute_infonce_loss(pos_features, neg_features)
            
            total_loss += batch_loss
            valid_batches += 1
        
        # 返回平均损失
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=student_pred.device, requires_grad=True)
    
    def _compute_infonce_loss(self, pos_features, neg_features):
        num_pos = pos_features.shape[0]
        
        pos_sim_matrix = torch.mm(pos_features, pos_features.T)  # [N_pos, N_pos]
        
        pos_neg_sim = torch.mm(pos_features, neg_features.T)  # [N_pos, N_neg]
        
        total_loss = 0.0
        
        for i in range(num_pos):
            pos_sims = torch.cat([pos_sim_matrix[i, :i], pos_sim_matrix[i, i+1:]])
            
            neg_sims = pos_neg_sim[i, :]
            
            if len(pos_sims) == 0:
                continue
            
            pos_exp = torch.exp(pos_sims).sum()
            
            all_sims = torch.cat([pos_sims, neg_sims])
            all_exp = torch.exp(all_sims).sum()
            
            if all_exp > 0:
                loss = -torch.log(pos_exp / (all_exp + 1e-8) + 1e-8)
                total_loss += loss
        
        return total_loss / max(num_pos, 1)

class ImitationLoss(nn.Module):
    def __init__(self, omega_KL=0.5, lambda_weights=None):
        super(ImitationLoss, self).__init__()
        self.omega_KL = omega_KL
        self.lambda_weights = lambda_weights if lambda_weights is not None else [1.0, 1.0, 1.0, 1.0, 1.0]
        
    def forward(self, student_pred, teacher_pred, targets, student_features=None, teacher_features=None):
        # KL散度损失
        L_KL= F.kl_div(F.log_softmax(student_pred, dim=1), F.softmax(teacher_pred, dim=1), reduction='batchmean')
        
        # 特征对齐损失
        L_mimic = 0.0
        if student_features is not None and teacher_features is not None:
            assert len(student_features) == len(teacher_features) == len(self.lambda_weights), \
                f"特征数量不匹配: student={len(student_features)}, teacher={len(teacher_features)}, weights={len(self.lambda_weights)}"
            
            for i, (s_feat, t_feat, lambda_l) in enumerate(zip(student_features, teacher_features, self.lambda_weights)):
                if s_feat.shape != t_feat.shape:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=True)
                
                l2_loss = torch.norm(s_feat - t_feat, p=2, dim=(2, 3)).pow(2).mean()
                L_mimic += lambda_l * l2_loss
                del s_feat, t_feat
                torch.cuda.empty_cache()
        else:
            L_mimic = F.mse_loss(student_pred, teacher_pred)
        
        total_loss = (1 - self.omega_KL) * L_mimic + self.omega_KL * L_KL
        
        return total_loss, L_mimic, L_KL


class CurriculumScheduler:
    def __init__(self, total_epochs, T=None):
        self.total_epochs = total_epochs
        self.T = T or int(0.6 * total_epochs)
        
    def get_training_phase(self, epoch):
        if epoch < self.T:
            return "imitation"
        else:
            return "preference"
    
    def get_curriculum_weight(self, epoch):
        t = epoch
        T = self.T
        
        if t < T:
            omega_KL = 0.5 * (1 + math.cos(math.pi * max(0, T - t) / T))
            
            w_mimic = 1.0 - omega_KL
            w_kl = omega_KL
            w_contrast = 0.0
            
        else:
            w_mimic = 0.0 
            w_kl = 0.0 
            w_contrast = 1.0 
            
        return w_mimic, w_kl, w_contrast
    
    def get_learning_focus(self, epoch):
        phase = self.get_training_phase(epoch)
        if phase == "imitation":
            return {
                'focus': '特征对齐和概率分布匹配',
                'strategy': '强化底层特征提取能力，学习教师模型的分割掩膜概率分布'
            }
        else:
            return {
                'focus': '偏好蒸馏和对比优化',
                'strategy': '基于对比相似性的偏好蒸馏机制，减少假阳性预测'
            }


def train_one_epoch_advanced(train_loader, teacher_model, student_model, 
                            criterion, imitation_criterion, contrastive_criterion,
                            optimizer, scheduler, epoch, logger, config, 
                            curriculum_scheduler, scaler=None):

    teacher_model.eval()
    student_model.train()

    loss_list = []
    seg_loss_list = []
    imitation_loss_list = []
    contrastive_loss_list = []
    l_mimic_list = []
    l_kl_list = []
    
    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        phase = curriculum_scheduler.get_training_phase(epoch)
        w_mimic, w_kl, w_contrast = curriculum_scheduler.get_curriculum_weight(epoch)
        
        if config.amp:
            with autocast():
                if phase == "imitation":
                    with torch.no_grad():
                        teacher_pred, teacher_features = teacher_model(images, return_features=True)
                    student_pred, student_features = student_model(images, return_features=True)
                    # with torch.no_grad():
                    #     teacher_pred = teacher_model(images, return_features=False)
                    # student_pred = student_model(images, return_features=False)
                    # student_features = None
                    # teacher_features = None 
                    
                    # 计算损失
                    seg_loss = criterion(student_pred, targets)
                    imitation_loss, L_mimic, L_KL = imitation_criterion(
                        student_pred, teacher_pred, student_features, teacher_features
                    )
                    total_loss = seg_loss + imitation_loss
                    
                    contrastive_loss = torch.tensor(0.0, device=images.device)
                else:
                    with torch.no_grad():
                        teacher_pred = teacher_model(images, return_features=False)
                    student_pred = student_model(images, return_features=False)
                    
                    # 计算损失
                    seg_loss = criterion(student_pred, targets)
                    contrastive_loss = contrastive_criterion(student_pred, teacher_pred, targets)
                    
                    # L2正则化
                    rho = 0.3
                    l2_reg = sum(torch.norm(param, p=2) for param in student_model.parameters())
                    
                    total_loss = seg_loss + contrastive_loss + rho * l2_reg
                    
                    imitation_loss, L_mimic, L_KL = imitation_criterion(
                        student_pred, teacher_pred, None, None
                    )
                
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if phase == "imitation":
                with torch.no_grad():
                    teacher_pred, teacher_features = teacher_model(images, return_features=True)
                student_pred, student_features = student_model(images, return_features=True)
                # with torch.no_grad():
                #     teacher_pred = teacher_model(images, return_features=False)
                # student_pred = student_model(images, return_features=False)
                # student_features = None
                # teacher_features = None
                
                # 计算损失
                seg_loss = criterion(student_pred, targets)
                imitation_loss, L_mimic, L_KL = imitation_criterion(
                    student_pred, teacher_pred, student_features, teacher_features
                )
                total_loss = seg_loss + imitation_loss
                
                contrastive_loss = torch.tensor(0.0, device=images.device)
                
            else:
                with torch.no_grad():
                    teacher_pred = teacher_model(images, return_features=False)
                student_pred = student_model(images, return_features=False)
                
                # 计算损失
                seg_loss = criterion(student_pred, targets)
                contrastive_loss = contrastive_criterion(student_pred, teacher_pred, targets)
                
                # L2正则化
                rho = 0.3
                l2_reg = sum(torch.norm(param, p=2) for param in student_model.parameters())
                
                total_loss = seg_loss + contrastive_loss + rho * l2_reg

                imitation_loss, L_mimic, L_KL = imitation_criterion(
                    student_pred, teacher_pred, None, None
                )
            
            total_loss.backward()
            optimizer.step()
        
        loss_list.append(total_loss.item())
        seg_loss_list.append(seg_loss.item())
        imitation_loss_list.append(imitation_loss.item())
        contrastive_loss_list.append(contrastive_loss.item())
        l_mimic_list.append(L_mimic.item() if hasattr(L_mimic, 'item') else L_mimic)
        l_kl_list.append(L_KL.item() if hasattr(L_KL, 'item') else L_KL)
        
        # 打印日志
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            learning_focus = curriculum_scheduler.get_learning_focus(epoch)
            
            log_info = f'train: epoch {epoch}, iter:{iter}, phase: {phase}, total_loss: {np.mean(loss_list):.4f}, ' \
                      f'seg_loss: {np.mean(seg_loss_list):.4f}, imitation_loss: {np.mean(imitation_loss_list):.4f}, ' \
                      f'contrastive_loss: {np.mean(contrastive_loss_list):.4f}'
            
            if phase == "imitation":
                log_info += f', L_mimic: {np.mean(l_mimic_list):.4f}, L_KL: {np.mean(l_kl_list):.4f}'
            
            log_info += f', w_mimic: {w_mimic:.3f}, w_kl: {w_kl:.3f}, w_contrast: {w_contrast:.3f}'
            log_info += f', lr: {now_lr}'
            
            print(log_info)
            logger.info(log_info)
            logger.info(f'Learning Focus: {learning_focus["focus"]} - {learning_focus["strategy"]}')
    
    scheduler.step()


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    teacher_model_path = os.path.join('results/malunet_Kvasir-SEG_Saturday_03_May_2025_02h_59m_24s/checkpoints/best-epoch288-loss0.4710.pth')  # 教师模型路径
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train_advanced', log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]  # 只使用显存占用较少的GPU
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    # 根据数据集类型选择相应的数据加载器
    if config.datasets == 'CVC-ClinicDB':
       
        train_dataset = CVC_datasets(config.data_path, config, train=True)
        val_dataset = CVC_datasets(config.data_path, config, train=False)
    else:
        train_dataset = NPY_datasets(config.data_path, config, train=True)
        val_dataset = NPY_datasets(config.data_path, config, train=False)
    
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Preparing Models----------#')
    model_cfg = config.model_config
    
    teacher_model = MALUNet(num_classes=model_cfg['num_classes'], 
                            input_channels=model_cfg['input_channels'], 
                            c_list=model_cfg['c_list'], 
                            split_att=model_cfg['split_att'],
                            bridge=model_cfg['bridge'])
    
    student_model = MAUNet(num_classes=model_cfg['num_classes'], 
                           input_channels=model_cfg['input_channels'], 
                           c_list=model_cfg['c_list'], 
                           bridge=model_cfg['bridge'])
    
    if os.path.exists(teacher_model_path):
        teacher_weights = torch.load(teacher_model_path, map_location=torch.device('cpu'))
        teacher_model.load_state_dict(teacher_weights)
        logger.info(f'Loaded teacher model from {teacher_model_path}')
    else:
        logger.info('Teacher model not found, using randomly initialized weights')
    
    teacher_model = torch.nn.DataParallel(teacher_model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    student_model = torch.nn.DataParallel(student_model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Preparing loss, opt, sch and amp----------#')
    criterion = config.criterion
    imitation_criterion = ImitationLoss()
    contrastive_criterion = ContrastiveLoss(temperature=0.1, max_pos_samples=256, max_neg_samples=512)
    if config.epochs <= 10:
        curriculum_scheduler = CurriculumScheduler(config.epochs, T=int(1.0 * config.epochs))
    else:
        curriculum_scheduler = CurriculumScheduler(config.epochs, T=max(int(0.8 * config.epochs), 8))
    
    optimizer = get_optimizer(config, student_model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        student_model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    print('#----------Advanced Training with Imitation and Contrastive Learning----------#')
    print(f'Curriculum Scheduler: total_epochs={config.epochs}, T={curriculum_scheduler.T}')
    print(f'Phase transition: epochs 1-{curriculum_scheduler.T-1} = imitation, epochs {curriculum_scheduler.T}-{config.epochs} = preference')
    
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        train_one_epoch_advanced(
            train_loader,
            teacher_model,
            student_model,
            criterion,
            imitation_criterion,
            contrastive_criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            curriculum_scheduler,
            scaler=scaler
        )

        loss = val_one_epoch(
                val_loader,
                student_model,
                criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            torch.save(student_model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': student_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        student_model.module.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                student_model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)

# nohup python -u train_new.py > train_new.log 2>&1 &
# python /home/lanping/newdisk/LanpingProject/MALUNet/train_new2.py