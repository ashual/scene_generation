import os

import torch
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

from scene_generation.data import imagenet_deprocess_batch
from scene_generation.discriminators import AcCropDiscriminator, define_mask_D, define_D
from scene_generation.losses import get_gan_losses, GANLoss, VGGLoss
from scene_generation.model import Model
from scene_generation.utils import LossManager


class Trainer:
    def __init__(self, args, vocab, checkpoint):
        self.vocab = vocab
        self.args = args
        self.num_obj = len(vocab['object_to_idx'])
        print(args.output_dir)
        self.writer = SummaryWriter(args.output_dir)
        self.colors = torch.randint(0, 256, [self.num_obj, 3]).float()
        self.gan_g_loss, self.gan_d_loss = get_gan_losses(args.gan_loss_type)

        self.init_generator(args, checkpoint)
        self.init_image_discriminator(args, checkpoint)
        self.init_obj_discriminator(args, checkpoint)
        self.init_mask_discriminator(args, checkpoint)

    def init_generator(self, args, checkpoint):
        if args.restore_from_checkpoint:
            model_kwargs = checkpoint['model_kwargs']
        else:
            model_kwargs = {
                'vocab': self.vocab,
                'image_size': args.image_size,
                'embedding_dim': args.embedding_dim,
                'gconv_dim': args.gconv_dim,
                'gconv_hidden_dim': args.gconv_hidden_dim,
                'gconv_num_layers': args.gconv_num_layers,
                'mlp_normalization': args.mlp_normalization,
                'appearance_normalization': args.appearance_normalization,
                'activation': args.activation,
                'mask_size': args.mask_size,
                'n_downsample_global': args.n_downsample_global,
                'box_dim': args.box_dim,
                'use_attributes': args.use_attributes,
                'box_noise_dim': args.box_noise_dim,
                'mask_noise_dim': args.mask_noise_dim,
                'pool_size': args.pool_size,
                'rep_size': args.rep_size,
            }
            checkpoint['model_kwargs'] = model_kwargs
        self.model = model = Model(**model_kwargs).to('cuda')
        # model.type(torch.cuda.FloatTensor)

        self.criterionVGG = VGGLoss() if args.vgg_features_weight > 0 else None
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionGAN = GANLoss(use_lsgan=not args.no_lsgan, tensor=torch.cuda.FloatTensor)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    def init_obj_discriminator(self, args, checkpoint):
        obj_discriminator, d_obj_kwargs, optimizer_d_obj = None, {}, None
        if args.d_obj_weight > 0:
            if args.restore_from_checkpoint:
                d_obj_kwargs = checkpoint['d_obj_kwargs']
            else:
                d_obj_kwargs = {
                    'vocab': self.vocab,
                    'arch': args.d_obj_arch,
                    'normalization': args.d_normalization,
                    'activation': args.d_activation,
                    'padding': args.d_padding,
                    'object_size': args.crop_size,
                }
                checkpoint['d_obj_kwargs'] = d_obj_kwargs
            obj_discriminator = AcCropDiscriminator(**d_obj_kwargs).to('cuda')
            # obj_discriminator.type(torch.cuda.FloatTensor)
            obj_discriminator.train()
            optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(), lr=args.learning_rate,
                                               betas=(args.beta1, 0.999))
        self.obj_discriminator = obj_discriminator
        self.optimizer_d_obj = optimizer_d_obj

    def init_mask_discriminator(self, args, checkpoint):
        mask_discriminator, d_mask_kwargs, optimizer_d_mask = None, {}, None
        if args.d_mask_weight > 0:
            if args.restore_from_checkpoint:
                d_mask_kwargs = checkpoint['d_mask_kwargs']
            else:
                use_sigmoid = args.no_lsgan
                netD_input_nc = 1
                d_mask_kwargs = {
                    'input_nc': netD_input_nc,
                    'ndf': args.ndf_mask,
                    'n_layers_D': args.n_layers_D_mask,
                    'norm': args.norm_D_mask,
                    'use_sigmoid': use_sigmoid,
                    'num_D': args.num_D_mask,
                    'num_objects': self.num_obj
                }
                checkpoint['d_mask_kwargs'] = d_mask_kwargs
            mask_discriminator = define_mask_D(**d_mask_kwargs).to('cuda')
            # mask_discriminator.type(torch.cuda.FloatTensor)
            mask_discriminator.train()
            optimizer_d_mask = torch.optim.Adam(mask_discriminator.parameters(), lr=args.mask_learning_rate,
                                                betas=(args.beta1, 0.999))
        self.mask_discriminator = mask_discriminator
        self.optimizer_d_mask = optimizer_d_mask

    def init_image_discriminator(self, args, checkpoint):
        if args.d_img_weight == 0:
            self.netD = None
            self.optimizer_d_img = None
            return
        use_sigmoid = args.no_lsgan
        netD_input_nc = self.num_obj + args.rep_size + args.output_nc
        if args.restore_from_checkpoint:
            d_img_kwargs = checkpoint['d_img_kwargs']
        else:
            d_img_kwargs = {
                'input_nc': netD_input_nc,
                'ndf': args.ndf,
                'n_layers_D': args.n_layers_D,
                'norm': args.norm_D,
                'use_sigmoid': use_sigmoid,
                'num_D': args.num_D,
            }
            checkpoint['d_img_kwargs'] = d_img_kwargs
        self.netD = netD = define_D(**d_img_kwargs).to('cuda')
        # netD.type(torch.cuda.FloatTensor)
        netD.train()
        self.optimizer_d_img = torch.optim.Adam(list(netD.parameters()), lr=args.learning_rate,
                                                betas=(args.beta1, 0.999))

    def restore_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optim_state'])

        if self.obj_discriminator is not None:
            self.obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
            self.optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

        if self.mask_discriminator is not None:
            self.mask_discriminator.load_state_dict(checkpoint['d_mask_state'])
            self.optimizer_d_mask.load_state_dict(checkpoint['d_mask_optim_state'])

        if self.netD is not None:
            self.netD.load_state_dict(checkpoint['d_img_state'])
            self.optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

    def save_checkpoint(self, checkpoint, t, args, epoch, train_results, val_results):
        print('checking on train')
        index = int(t / args.print_every)

        t_avg_iou, t_inception_mean, t_inception_std, _ = train_results
        self.writer.add_scalar('checkpoint/{}'.format('train_iou'), t_avg_iou, index)
        self.writer.add_scalar('checkpoint/{}'.format('train_inception_mean'), t_inception_mean, index)
        self.writer.add_scalar('checkpoint/{}'.format('train_inception_std'), t_inception_std, index)
        checkpoint['checkpoint_ts'].append(t)
        checkpoint['train_inception'].append(t_inception_mean)

        print('checking on val')
        val_avg_iou, val_inception_mean, val_inception_std, _ = val_results
        self.writer.add_scalar('checkpoint/{}'.format('val_iou'), val_avg_iou, index)
        self.writer.add_scalar('checkpoint/{}'.format('val_inception_mean'), val_inception_mean, index)
        self.writer.add_scalar('checkpoint/{}'.format('val_inception_std'), val_inception_std, index)
        checkpoint['val_inception'].append(val_inception_mean)

        print('train iou: ', t_avg_iou)
        print('val iou: ', val_avg_iou)

        if self.obj_discriminator is not None:
            checkpoint['d_obj_state'] = self.obj_discriminator.state_dict()
            checkpoint['d_obj_optim_state'] = self.optimizer_d_obj.state_dict()

        if self.mask_discriminator is not None:
            checkpoint['d_mask_state'] = self.mask_discriminator.state_dict()
            checkpoint['d_mask_optim_state'] = self.optimizer_d_mask.state_dict()

        if self.netD is not None:
            checkpoint['d_img_state'] = self.netD.state_dict()
            checkpoint['d_img_optim_state'] = self.optimizer_d_img.state_dict()

        checkpoint['model_state'] = self.model.state_dict()
        checkpoint['optim_state'] = self.optimizer.state_dict()

        if len(checkpoint['best_t']) == 0 or max(checkpoint['val_inception']) < val_inception_mean:
            checkpoint['best_t'].append(t)
            checkpoint['d_obj_best_state'] = checkpoint['d_obj_state']
            checkpoint['d_obj_optim_best_state'] = checkpoint['d_obj_optim_state']
            checkpoint['d_mask_best_state'] = checkpoint['d_mask_state']
            checkpoint['d_mask_optim_best_state'] = checkpoint['d_mask_optim_state']
            checkpoint['d_img_best_state'] = checkpoint['d_img_state']
            checkpoint['d_img_optim_best_state'] = checkpoint['d_img_optim_state']
            checkpoint['model_best_state'] = checkpoint['model_state']
            checkpoint['optim_best_state'] = checkpoint['optim_state']

        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = epoch
        checkpoint_path = os.path.join(args.output_dir, '%s_with_model.pt' % args.checkpoint_name)
        print('Saving checkpoint to ', checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    def train_generator(self, imgs, imgs_pred, masks, masks_pred, layout,
                        objs, boxes, boxes_pred, obj_to_img, use_gt):
        args = self.args
        self.generator_losses = LossManager()

        if use_gt:
            if args.l1_pixel_loss_weight > 0:
                l1_pixel_loss = F.l1_loss(imgs_pred, imgs)
                self.generator_losses.add_loss(l1_pixel_loss, 'L1_pixel_loss', args.l1_pixel_loss_weight)

            loss_bbox = F.mse_loss(boxes_pred, boxes)
            self.generator_losses.add_loss(loss_bbox, 'bbox_pred', args.bbox_pred_loss_weight)

        # VGG feature matching loss
        if self.criterionVGG is not None:
            loss_G_VGG = self.criterionVGG(imgs_pred, imgs)
            self.generator_losses.add_loss(loss_G_VGG, 'g_vgg', args.vgg_features_weight)

        scores_fake, ac_loss, g_fake_crops = self.obj_discriminator(imgs_pred, objs, boxes, obj_to_img)
        self.generator_losses.add_loss(ac_loss, 'ac_loss', args.ac_loss_weight)
        weight = args.d_obj_weight
        self.generator_losses.add_loss(self.gan_g_loss(scores_fake), 'g_gan_obj_loss', weight)

        if self.mask_discriminator is not None:
            O, _, mask_size = masks_pred.shape
            one_hot_size = (O, self.num_obj)
            one_hot_obj = torch.zeros(one_hot_size, dtype=masks_pred.dtype, device=masks_pred.device)
            one_hot_obj = one_hot_obj.scatter_(1, objs.view(-1, 1).long(), 1.0)

            scores_fake = self.mask_discriminator(masks_pred.unsqueeze(1), one_hot_obj)
            mask_loss = self.criterionGAN(scores_fake, True)
            self.generator_losses.add_loss(mask_loss, 'g_gan_mask_obj_loss', args.d_mask_weight)

            # GAN feature matching loss
            if args.d_mask_features_weight > 0:
                scores_real = self.mask_discriminator(masks.float().unsqueeze(1), one_hot_obj)
                loss_mask_feat = self.calculate_features_loss(scores_fake, scores_real)
                self.generator_losses.add_loss(loss_mask_feat, 'g_mask_features_loss', args.d_mask_features_weight)

        if self.netD is not None:
            # Train textures
            pred_real = self.netD.forward(torch.cat((layout, imgs), dim=1))

            # Train image generation
            match_layout = layout.detach()
            img_pred_fake = self.netD.forward(torch.cat((match_layout, imgs_pred), dim=1))
            g_gan_img_loss = self.criterionGAN(img_pred_fake, True)
            self.generator_losses.add_loss(g_gan_img_loss, 'g_gan_img_loss', args.d_img_weight)

            if args.d_img_features_weight > 0:
                loss_g_gan_feat_img = self.calculate_features_loss(img_pred_fake, pred_real)
                self.generator_losses.add_loss(loss_g_gan_feat_img,
                                               'g_gan_features_loss_img', args.d_img_features_weight)

        self.generator_losses.all_losses['total_loss'] = self.generator_losses.total_loss.item()

        self.optimizer.zero_grad()
        self.generator_losses.total_loss.backward()
        self.optimizer.step()

    def train_obj_discriminator(self, imgs, imgs_pred, objs, boxes, boxes_pred, obj_to_img):
        if self.obj_discriminator is not None:
            self.d_obj_losses = d_obj_losses = LossManager()
            scores_fake, ac_loss_fake, self.d_fake_crops = self.obj_discriminator(imgs_pred, objs, boxes_pred,
                                                                                  obj_to_img)
            scores_real, ac_loss_real, self.d_real_crops = self.obj_discriminator(imgs, objs, boxes, obj_to_img)

            d_obj_gan_loss = self.gan_d_loss(scores_real, scores_fake)
            d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss', 0.5)
            d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
            d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

            self.optimizer_d_obj.zero_grad()
            d_obj_losses.total_loss.backward()
            self.optimizer_d_obj.step()

    def train_mask_discriminator(self, masks, masks_pred, objs):
        if self.mask_discriminator is not None:
            self.d_mask_losses = d_mask_losses = LossManager()

            O, _, mask_size = masks_pred.shape
            one_hot_size = (O, self.num_obj)
            one_hot_obj = torch.zeros(one_hot_size, dtype=masks_pred.dtype, device=masks_pred.device)
            one_hot_obj = one_hot_obj.scatter_(1, objs.view(-1, 1).long(), 1.0)

            scores_fake = self.mask_discriminator(masks_pred.unsqueeze(1), one_hot_obj)
            scores_real = self.mask_discriminator(masks.float().unsqueeze(1), one_hot_obj)

            fake_loss = self.criterionGAN(scores_fake, False)
            real_loss = self.criterionGAN(scores_real, True)
            d_mask_losses.add_loss(fake_loss, 'fake_loss', 0.5)
            d_mask_losses.add_loss(real_loss, 'real_loss', 0.5)

            self.optimizer_d_mask.zero_grad()
            d_mask_losses.total_loss.backward()
            self.optimizer_d_mask.step()

    def train_image_discriminator(self, imgs, imgs_pred, layout, layout_wrong):
        if self.netD is not None:
            self.d_img_losses = d_img_losses = LossManager()
            # Fake Detection and Loss
            alpha = (1 / 2) * (.5)

            # Fake images, Real layout
            pred_fake_pool_img = self.discriminate(layout, imgs_pred)
            loss_d_fake_img = self.criterionGAN(pred_fake_pool_img, False)
            d_img_losses.add_loss(loss_d_fake_img, 'fake_image_loss', alpha)

            # Real images, Right layout Wrong textures
            pred_wrong_pool_img = self.discriminate(layout_wrong, imgs)
            loss_d_wrong_texture = self.criterionGAN(pred_wrong_pool_img, False)
            d_img_losses.add_loss(loss_d_wrong_texture, 'wrong_texture_loss', alpha)

            # Real Detection and Loss
            pred_real = self.discriminate(layout, imgs)
            loss_D_real = self.criterionGAN(pred_real, True)
            d_img_losses.add_loss(loss_D_real, 'd_img_gan_real_loss', 0.5)

            self.optimizer_d_img.zero_grad()
            d_img_losses.total_loss.backward()
            self.optimizer_d_img.step()

    def discriminate(self, input_label, test_image):
        input_concat = torch.cat((input_label, test_image), dim=1)
        return self.netD.forward(input_concat)

    def calculate_features_loss(self, pred_fake, pred_real):
        loss_G_GAN_Feat = 0
        nums_d = len(pred_fake)
        feat_weights = 4.0 / len(pred_fake[0])
        D_weights = 1.0 / nums_d
        for i in range(nums_d):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                                   self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        return loss_G_GAN_Feat

    def write_losses(self, checkpoint, t):
        writer = self.writer
        index = int(t / self.args.print_every)
        print('t = %d / %d' % (t, self.args.num_iterations))
        for name, val in self.generator_losses.items():
            print(' G [%s]: %.4f' % (name, val))
            checkpoint['losses'][name].append(val)
            writer.add_scalar('g_loss/{}'.format(name), val, index)
        checkpoint['losses_ts'].append(t)

        if self.obj_discriminator is not None:
            for name, val in self.d_obj_losses.items():
                print(' D_obj [%s]: %.4f' % (name, val))
                checkpoint['d_losses'][name].append(val)
                writer.add_scalar('d_obj_loss/{}'.format(name), val, index)

        if self.mask_discriminator is not None:
            for name, val in self.d_mask_losses.items():
                print(' D_mask [%s]: %.4f' % (name, val))
                checkpoint['d_losses'][name].append(val)
                writer.add_scalar('d_mask_loss/{}'.format(name), val, index)

        if self.netD is not None:
            for name, val in self.d_img_losses.items():
                print(' D_img [%s]: %.4f' % (name, val))
                checkpoint['d_losses'][name].append(val)
                writer.add_scalar('d_img_loss/{}'.format(name), val, index)

    def write_images(self, t, imgs, imgs_pred, layout_one_hot, layout_pred_one_hot):
        writer = self.writer
        index = int(t / self.args.print_every)
        imgs_print = imagenet_deprocess_batch(imgs)
        writer.add_image('img/real', torchvision.utils.make_grid(imgs_print, normalize=True, scale_each=True), index)
        if imgs_pred is not None:
            imgs_pred_print = imagenet_deprocess_batch(imgs_pred)
            writer.add_image('img/pred', torchvision.utils.make_grid(imgs_pred_print, normalize=True, scale_each=True),
                             index)
        if self.obj_discriminator is not None:
            d_real_crops_print = imagenet_deprocess_batch(self.d_real_crops)
            writer.add_image('objs/d_real',
                             torchvision.utils.make_grid(d_real_crops_print, normalize=True, scale_each=True), index)
            g_fake_crops_print = imagenet_deprocess_batch(self.d_fake_crops)
            writer.add_image('objs/g_fake',
                             torchvision.utils.make_grid(g_fake_crops_print, normalize=True, scale_each=True), index)
        layout_one_hot_3d = self.one_hot_to_rgb(layout_one_hot)
        writer.add_image('img/layout',
                         torchvision.utils.make_grid(layout_one_hot_3d, normalize=True, scale_each=True), index)
        layout_pred_one_hot_3d = self.one_hot_to_rgb(layout_pred_one_hot)
        writer.add_image('img/layout_pred',
                         torchvision.utils.make_grid(layout_pred_one_hot_3d, normalize=True, scale_each=True),
                         index)

    def one_hot_to_rgb(self, one_hot):
        one_hot_3d = torch.einsum('abcd,be->aecd', [one_hot.cpu(), self.colors])
        one_hot_3d *= (255.0 / one_hot_3d.max())
        return one_hot_3d
