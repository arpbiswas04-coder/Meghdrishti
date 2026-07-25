import os
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import time
import argparse
import yaml
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.models as models
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torch.cuda.amp import autocast, GradScaler
from data_manager import TrainDataset
from data.sen12mscr_dataset import SEN12MSCRDataset
from models.gen.SPANet import Generator
from models.dis.dis import Discriminator
from models.cloud_head import CloudDetectionHead
import utils

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(10)]) # up to conv2_2 (layer 9)
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(10, 17)]) # up to conv3_3 (layer 16)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x3 = x[:, :3, :, :]
        y3 = y[:, :3, :, :]
        h_x1 = self.slice1(x3)
        h_y1 = self.slice1(y3)
        h_x2 = self.slice2(h_x1)
        h_y2 = self.slice2(h_y1)
        return F.l1_loss(h_x1, h_y1) + F.l1_loss(h_x2, h_y2)

def spectral_consistency_loss(pred, target, mask):
    non_cloud = 1.0 - mask
    if non_cloud.sum() < 1:
        return torch.tensor(0.0).to(pred.device)
    
    pred_nc = pred * non_cloud
    target_nc = target * non_cloud
    
    mean_pred = pred_nc.sum(dim=(2,3)) / (non_cloud.sum(dim=(2,3)) + 1e-8)
    mean_target = target_nc.sum(dim=(2,3)) / (non_cloud.sum(dim=(2,3)) + 1e-8)
    
    var_pred = ((pred_nc - mean_pred.unsqueeze(-1).unsqueeze(-1))**2 * non_cloud).sum(dim=(2,3)) / (non_cloud.sum(dim=(2,3)) + 1e-8)
    var_target = ((target_nc - mean_target.unsqueeze(-1).unsqueeze(-1))**2 * non_cloud).sum(dim=(2,3)) / (non_cloud.sum(dim=(2,3)) + 1e-8)
    
    std_pred = torch.sqrt(var_pred + 1e-8)
    std_target = torch.sqrt(var_target + 1e-8)
    
    return F.l1_loss(mean_pred, mean_target) + F.l1_loss(std_pred, std_target)

def calculate_psnr_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
    p, s = 0.0, 0.0
    for i in range(pred_np.shape[0]):
        p_img = pred_np[i]
        t_img = target_np[i]
        
        # handle different channel sizes by taking first 3
        if p_img.shape[-1] > 3:
            p_img = p_img[:, :, :3]
            t_img = t_img[:, :, :3]
            
        p += psnr(t_img, p_img, data_range=t_img.max() - t_img.min())
        s += ssim(t_img, p_img, data_range=t_img.max() - t_img.min(), channel_axis=2)
    return p / pred_np.shape[0], s / pred_np.shape[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rice', choices=['rice', 'sen12mscr'])
    parser.add_argument('--use_sar', action='store_true', default=False)
    parser.add_argument('--in_channels', type=int, default=-1)
    parser.add_argument('--max_samples', type=int, default=3000)
    parser.add_argument('--dataroot', type=str, default='./pretrained_models')
    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--n_epochs', type=int, default=0)
    parser.add_argument('--pretrained', type=str, default=None)
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Load default config
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = argparse.Namespace(**config)
    
    # Override with args
    if args.in_channels == -1:
        args.in_channels = 6 if args.use_sar else 4
        
    config.in_ch = args.in_channels
    config.out_ch = 4
    if args.n_epochs > 0:
        config.epoch = args.n_epochs
    config.datasets_dir = args.dataroot
    if config.datasets_dir.endswith('RICE_DATASET') or config.datasets_dir.endswith('pretrained_models'):
        config.datasets_dir = os.path.join(config.datasets_dir, 'RICE1')
    config.out_dir = os.path.join(config.out_dir, args.name)
    os.makedirs(config.out_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    print('===> Loading datasets')
    if args.dataset == 'rice':
        dataset = TrainDataset(config)
    else:
        dataset = SEN12MSCRDataset(args.dataroot, split='train', max_samples=args.max_samples)
        
    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=4, shuffle=True, pin_memory=False, persistent_workers=False)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=0, batch_size=4, shuffle=False, pin_memory=False, persistent_workers=False)

    print('===> Loading models')
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        config.gpu_ids = []
    gen = Generator(gpu_ids=config.gpu_ids, in_ch=config.in_ch, out_ch=config.out_ch).to(device)
    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids).to(device)
    head = CloudDetectionHead(in_channels=config.in_ch).to(device)
    
    # Generator init
    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('InstanceNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    gen.apply(init_weights)

    # Discriminator pretrained loading
    matched_dis_layers = 0
    dis_pretrained_path = os.path.join(config.datasets_dir, 'dis_model_epoch_200.pth')
    if args.pretrained:
        dis_pretrained_path = args.pretrained.replace('gen_model', 'dis_model')
        
    if os.path.exists(dis_pretrained_path):
        state_dict = torch.load(dis_pretrained_path, map_location=device, weights_only=True)
        model_dict = dis.state_dict()
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                matched_dis_layers += 1
        dis.load_state_dict(model_dict)

    opt_gen = optim.Adam(list(gen.parameters()) + list(head.parameters()), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=1e-5)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=1e-5)

    scaler = GradScaler()

    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    criterionSoftplus = nn.Softplus().to(device)
    vgg_loss_fn = VGGLoss().to(device)

    cloudsen_path = os.path.join('./data/CloudSEN12')
    has_cloudsen = os.path.exists(os.path.join(cloudsen_path, 'masks.npy'))
    if has_cloudsen:
        print("Loaded CloudSEN12+ for head training.")
        csen_images = torch.from_numpy(np.load(os.path.join(cloudsen_path, 'images.npy'))).float()
        csen_masks = torch.from_numpy(np.load(os.path.join(cloudsen_path, 'masks.npy'))).float()

    best_psnr = 0.0
    total_iters = 0
    start_time = time.time()
    
    print("=======================================")
    print("VERIFICATION")
    print(f"Dataroot being used: {config.datasets_dir}")
    print(f"RICE version detected: {os.path.basename(config.datasets_dir)}")
    print(f"Cloudy folder name found: {getattr(config, 'detected_input_dir', 'N/A')}")
    print(f"Target folder name found: {getattr(config, 'detected_target_dir', 'N/A')}")
    print(f"Number of training pairs: {getattr(config, 'train_size', 0)}")
    print(f"Number of validation pairs: {getattr(config, 'val_size', 0)}")
    print(f"Generator: randomly initialized with kaiming_normal_")
    if matched_dis_layers > 0:
        print(f"Discriminator: pretrained weights loaded ({matched_dis_layers} layers matched)")
    else:
        print(f"Discriminator: randomly initialized if no match")
    print("=======================================")

    for epoch in range(1, config.epoch + 1):
        gen.train()
        head.train()
        dis.train()
        
        for iteration, batch in enumerate(training_data_loader, 1):
            total_iters += 1
            
            if isinstance(batch, dict):
                real_a = batch['input'].to(device)
                real_b = batch['target'].to(device)
                sar = batch['sar'].to(device)
                if args.use_sar:
                    real_a = torch.cat([real_a, sar], dim=1)
            else:
                real_a, real_b, _ = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            if total_iters == 1:
                print("Input stats: min=", real_a.min().item(),
                      "max=", real_a.max().item(),
                      "shape=", real_a.shape,
                      "device=", real_a.device)
                print("Target stats: min=", real_b.min().item(),
                      "max=", real_b.max().item())
            
            batchsize = real_a.size(0)
            
            with autocast():
                att, fake_b = gen(real_a)
                pred_mask = head(real_a)
                mask_sig = torch.sigmoid(pred_mask)

            ### Update D ###
            opt_dis.zero_grad()
            with autocast():
                fake_ab = torch.cat((real_a, fake_b.detach()), 1)
                pred_fake_1, pred_fake_2 = dis(fake_ab)
                loss_d_fake = (criterionSoftplus(pred_fake_1).mean() + criterionSoftplus(pred_fake_2).mean()) / 2.0

                real_ab = torch.cat((real_a, real_b), 1)
                pred_real_1, pred_real_2 = dis(real_ab)
                loss_d_real = (criterionSoftplus(-pred_real_1).mean() + criterionSoftplus(-pred_real_2).mean()) / 2.0

                loss_d = loss_d_fake + loss_d_real
                
            scaler.scale(loss_d).backward()
            if epoch % config.minimax == 0:
                scaler.step(opt_dis)

            ### Update G & Head ###
            opt_gen.zero_grad()
            with autocast():
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake_1, pred_fake_2 = dis(fake_ab)
                loss_g_gan = (criterionSoftplus(-pred_fake_1).mean() + criterionSoftplus(-pred_fake_2).mean()) / 2.0

                loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
                
                # Use generated mask as attention target
                att_target = mask_sig.detach()
                # If attention is multi-channel, match it
                if att.shape[1] == 4:
                    att_target = att_target.repeat(1, 4, 1, 1)
                elif att.shape[1] == 1:
                    pass
                loss_g_att = criterionMSE(att, att_target) * 0.1
                
                loss_perceptual = vgg_loss_fn(fake_b, real_b) * 0.1
                loss_spectral = spectral_consistency_loss(fake_b, real_b, mask_sig.detach()) * 0.05
                
                loss_g = loss_g_gan + loss_g_l1 + loss_g_att + loss_perceptual + loss_spectral

                # Cloud head loss
                head_loss = 0.0
                if has_cloudsen:
                    idx = torch.randint(0, len(csen_images), (batchsize,))
                    c_img = csen_images[idx].to(device)
                    c_img_4 = c_img[:, [1, 2, 3, 7]]
                    if config.in_ch == 6:
                        c_img_4 = torch.cat([c_img_4, torch.zeros(batchsize, 2, 256, 256).to(device)], dim=1)
                    c_mask = csen_masks[idx].to(device)
                    head_loss += F.binary_cross_entropy_with_logits(head(c_img_4), c_mask)
                
                pseudo_mask = (torch.abs(real_a[:, :3] - real_b[:, :3]).mean(dim=1, keepdim=True) > 0.2).float()
                head_loss += F.binary_cross_entropy_with_logits(pred_mask, pseudo_mask)

                total_loss_g = loss_g + head_loss

            scaler.scale(total_loss_g).backward()
            scaler.step(opt_gen)
            scaler.update()

            if total_iters % 10 == 0:
                p, s = calculate_psnr_ssim(fake_b, real_b)
                elapsed = time.time() - start_time
                iters_per_sec = total_iters / elapsed
                iters_left = (config.epoch * len(training_data_loader)) - total_iters
                eta_hrs = (iters_left / iters_per_sec) / 3600
                
                print(f"Epoch {epoch} [{iteration}/{len(training_data_loader)}] "
                      f"L_D: {loss_d.item():.3f} L_G: {loss_g.item():.3f} "
                      f"PSNR: {p:.2f} SSIM: {s:.3f} | ETA: {eta_hrs:.2f}h")

            if total_iters % 500 == 0:
                torch.cuda.empty_cache()
                gen.eval()
                val_psnr, val_ssim = 0.0, 0.0
                with torch.no_grad():
                    for val_batch in validation_data_loader:
                        if isinstance(val_batch, dict):
                            v_a = val_batch['input'].to(device)
                            v_b = val_batch['target'].to(device)
                            v_sar = val_batch['sar'].to(device)
                            if args.use_sar:
                                v_a = torch.cat([v_a, v_sar], dim=1)
                        else:
                            v_a, v_b = val_batch[0].to(device), val_batch[1].to(device)
                            
                        _, v_fake = gen(v_a)
                        vp, vs = calculate_psnr_ssim(v_fake, v_b)
                        val_psnr += vp
                        val_ssim += vs
                        
                val_psnr /= len(validation_data_loader)
                val_ssim /= len(validation_data_loader)
                print(f"===> Validation PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.3f}")
                
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(gen.state_dict(), 'checkpoints/best_psnr.pth')
                    print("Saved new best model!")
                if epoch % 1 == 0:
                    torch.save(gen.state_dict(), f"{config.out_dir}/gen_epoch_{epoch}.pth")
                    torch.save(dis.state_dict(), f"{config.out_dir}/dis_epoch_{epoch}.pth")
                gen.train()
                torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
