import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from models import MDTraj

ncols = 150
class Processor():
    def __init__(self, args):
        self.args = args
        if args.test_set == 9:
            from utils_sdd import DataSet_bytrajec2_sdd, DataLoader, L2forTest
            self.trainloader_gt = DataLoader(DataSet_bytrajec2_sdd(args,is_train=True), batch_size=1, shuffle=True, num_workers=0)
            self.testloader_gt = DataLoader(DataSet_bytrajec2_sdd(args,is_train=False), batch_size=1, shuffle=False, num_workers=0)
        else:
            from utils_ethucy import DataSet_bytrajec2_ethucy, DataLoader, L2forTest
            self.trainloader_gt = DataLoader(DataSet_bytrajec2_ethucy(args,is_train=True), batch_size=1, shuffle=True, num_workers=0)
            self.testloader_gt = DataLoader(DataSet_bytrajec2_ethucy(args,is_train=False), batch_size=1, shuffle=False, num_workers=0)
        
        self.lr=self.args.learning_rate
        self.L2forTest = L2forTest
        self.net = MDTraj(args)
        if self.args.phase == "train":
            print("self.args.phase",self.args.phase)
            self.net.train()
        else:
            self.net.eval()
        self.init_lr = self.args.learning_rate
        self.step_ratio = self.args.step_ratio
        self.lr_step=self.args.lr_step
        self.set_optimizer()
        self.epoch = 1
        self.load_model()
        self.net = self.net.cuda()


    def save_model(self,epoch):
        model_path = self.args.save_dir + '/' + self.args.train_model + '/'
        torch.save(self.net.state_dict(), model_path + 'MDTraj_%d.pth' % epoch)
        # if self.args.train_phase == 'gen_memory':
        #     torch.save(self.net.t_encoder.state_dict(), model_path + 't_encoder_%d.pth' % epoch)
        #     torch.save(self.net.s_encoder.state_dict(), model_path + 's_encoder_%d.pth' % epoch)
        #     torch.save(self.net.key_point_encoder.state_dict(), model_path + 'key_point_encoder_%d.pth' % epoch)
        #     torch.save(self.net.tf_decoder.state_dict(), model_path + 'tf_decoder_%d.pth' % epoch)
        # if self.args.train_phase == 'train_adaptor':
        #     torch.save(self.net.t_difference_encoder.state_dict(), model_path + 't_difference_encoder_%d.pth' % epoch)
        #     torch.save(self.net.s_difference_encoder.state_dict(), model_path + 's_difference_encoder_%d.pth' % epoch)
        #     torch.save(self.net.key_adaptor.state_dict(), model_path + 'key_adaptor_%d.pth' % epoch)
        #     torch.save(self.net.re_proj.state_dict(), model_path + 're_proj_%d.pth' % epoch)

        #     torch.save(self.net.tf_decoder.state_dict(), model_path + 'tf_decoder_%d.pth' % epoch)
        # if self.args.train_phase == 'train_diffusion':
        #     torch.save(self.net.my_diff.state_dict(), model_path + 'my_diff_%d.pth' % epoch)
    

    def load_model(self):
        epoch = self.args.load_model
        if not epoch > 0:
            return
        
        model_path = self.args.save_dir + '/' + self.args.train_model + '/'
        self.net.load_state_dict(torch.load(model_path + 'MDTraj_%d.pth' % epoch), strict=False)
        self.epoch = epoch

        for i in range(self.args.load_model):
            self.scheduler.step()
        # if self.args.train_phase == 'gen_memory':
        #     if self.args.phase == 'test':
        #         self.net.t_encoder.load_state_dict(torch.load(model_path + 't_encoder_%d.pth' % epoch))
        #         self.net.s_encoder.load_state_dict(torch.load(model_path + 's_encoder_%d.pth' % epoch))
        #         self.net.key_point_encoder.load_state_dict(torch.load(model_path + 'key_point_encoder_%d.pth' % epoch))
        #         self.net.tf_decoder.load_state_dict(torch.load(model_path + 'tf_decoder_%d.pth' % epoch))

        #     # print('load model:', model_path + 't_encoder, s_encoder, key_point_encoder, tf_decoder at epoch %d' % epoch)
        # if self.args.train_phase == 'train_adaptor':
        #     self.net.t_encoder.load_state_dict(torch.load(model_path + 't_encoder_%d.pth' % epoch))
        #     self.net.s_encoder.load_state_dict(torch.load(model_path + 's_encoder_%d.pth' % epoch))
        #     self.net.key_point_encoder.load_state_dict(torch.load(model_path + 'key_point_encoder_%d.pth' % epoch))
        #     self.net.tf_decoder.load_state_dict(torch.load(model_path + 'tf_decoder_%d.pth' % epoch))

        #     print('load model:', model_path + 't_encoder, s_encoder, key_point_encoder, tf_decoder at epoch %d' % epoch)
        #     if self.args.phase == 'test':
        #         self.net.t_difference_encoder.load_state_dict(torch.load(model_path + 't_difference_encoder_%d.pth' % epoch))
        #         self.net.s_difference_encoder.load_state_dict(torch.load(model_path + 's_difference_encoder_%d.pth' % epoch))
        #         self.net.key_adaptor.load_state_dict(torch.load(model_path + 'key_adaptor_%d.pth' % epoch))
        #         self.net.re_proj.load_state_dict(torch.load(model_path + 're_proj_%d.pth' % epoch))

                
        #         print('test phase, load t_difference_encoder, s_difference_encoder, key_adaptor, re_proj at epoch %d' % epoch)
                
        # if self.args.train_phase == 'train_diffusion':
        #     self.net.t_encoder.load_state_dict(torch.load(model_path + 't_encoder_%d.pth' % epoch))
        #     self.net.s_encoder.load_state_dict(torch.load(model_path + 's_encoder_%d.pth' % epoch))
        #     self.net.key_point_encoder.load_state_dict(torch.load(model_path + 'key_point_encoder_%d.pth' % epoch))
        #     self.net.tf_decoder.load_state_dict(torch.load(model_path + 'tf_decoder_%d.pth' % epoch))
        #     self.net.t_difference_encoder.load_state_dict(torch.load(model_path + 't_difference_encoder_%d.pth' % epoch))
        #     self.net.s_difference_encoder.load_state_dict(torch.load(model_path + 's_difference_encoder_%d.pth' % epoch))
        #     self.net.key_adaptor.load_state_dict(torch.load(model_path + 'key_adaptor_%d.pth' % epoch))
        #     self.net.re_proj.load_state_dict(torch.load(model_path + 're_proj_%d.pth' % epoch))


        #     print('load model:', model_path + 't_encoder, s_encoder, key_point_encoder, tf_decoder, t_difference_encoder, s_difference_encoder, key_adaptor at epoch %d' % epoch)

        #     if self.args.phase == 'test':
        #         self.net.my_diff.load_state_dict(torch.load(model_path + 'my_diff_%d.pth' % epoch))
        #         print('test phase, load my_diff at epoch %d' % epoch)

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max = self.args.num_epochs, eta_min=self.args.eta_min)

    def playtest(self):
        print('loading memory')
        self.load_memory(self.args.load_model)
        print('Testing begin')
        print('Model:', self.args.load_model)

        test_error, test_final_error = self.test_epoch(self.args.load_model)
        for k in test_error.keys():
            print('*** agent class: ', k, '***')
            print('ade=%.4f, fde=%.4f' % (test_error[k], test_final_error[k]))

    def playtrain(self):
        print('Training begin')
        if self.args.train_phase != 'gen_memory':
            print('loading memory')
            self.load_memory(self.args.load_model)
            
        test_error, test_final_error = 0,0
        for epoch in range(self.epoch, self.args.num_epochs+1):
            self.train_epoch(epoch)

            self.scheduler.step()
            if epoch == self.args.num_epochs:
                self.save_model(epoch)
            # if epoch == self.args.num_epochs or epoch % 1 == 0:
            #     test_error, test_final_error = self.test_epoch(epoch)
            #     for k in test_error.keys():
            #         print('*** type: ', k, '***')
            #         print('ade=%.4f, fde=%.4f' % (test_error[k], test_final_error[k]))
            if self.args.train_phase == 'gen_memory' and epoch == self.args.num_epochs:
                print('training gen_memory finished, updating memory !')
                self.generate_memory(epoch)

    def train_epoch(self, epoch):
        self.net.train()
        loss_epoch = 0

        # 使用tqdm创建一个进度条
        trainloader_gt_progress = tqdm(enumerate(self.trainloader_gt), total=len(self.trainloader_gt), ncols=ncols)

        for batch, (batch_abs_gt, batch_norm_gt, key_points, shift_values, max_values, batch_split, edge_pair) in trainloader_gt_progress:
            batch_abs_gt = batch_abs_gt[0].float().cuda()
            batch_norm_gt = batch_norm_gt[0].float().cuda()
            key_points = key_points[0].float().cuda()
            shift_values = shift_values[0].float().cuda()
            max_values = max_values[0].float().cuda()
            edge_pair = [x[0].long().cuda() for x in edge_pair]
            inputs_fw = batch_abs_gt, batch_norm_gt, key_points, shift_values, max_values, batch_split, edge_pair
            self.optimizer.zero_grad()

            tot_loss = self.net.forward(inputs_fw)

            trainloader_gt_progress.set_description('train-{}/{},epoch={},train_phase=<{}>,reg_loss={:.4f},loss_avg={:.4f}'.format(
                batch, len(self.trainloader_gt), epoch, self.args.train_phase,
                tot_loss.item(),
                loss_epoch / (batch + 1),
            ))

            loss_epoch += tot_loss.item()
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()

    def generate_memory(self, epoch):
        self.net.eval()
        self.memory_t = torch.Tensor().cuda() # memory is initialized by empty tensor
        self.memory_s = torch.Tensor().cuda()
        self.memory_key = torch.Tensor().cuda()
        # self.memory_y = torch.Tensor().cuda()
        self.complete_nbr = [] # 用于存储每个agent的完整邻居轨迹
        self.shift_values = torch.Tensor().cuda()
        self.max_values = torch.Tensor().cuda()

        with torch.no_grad():
            trainloader_gt_progress = tqdm(enumerate(self.trainloader_gt), total=len(self.trainloader_gt), ncols=ncols)
            for batch, (batch_abs_gt, batch_norm_gt, key_points, shift_values, max_values, batch_split, edge_pair) in trainloader_gt_progress:
                batch_abs_gt = batch_abs_gt[0].float().cuda()
                batch_norm_gt = batch_norm_gt[0].float().cuda()
                key_points = key_points[0].float().cuda()
                shift_values = shift_values[0].float().cuda()
                max_values = max_values[0].float().cuda()
                edge_pair = [x[0].long().cuda() for x in edge_pair]
                inputs_fw = batch_abs_gt, batch_norm_gt, key_points, shift_values, max_values, batch_split, edge_pair

                x_encoding, x_social, key_point_encoding = self.net.forward(inputs_fw, True)
                
                x_social_ = x_social.transpose(0, 1) # [N, 8, D]
                self.memory_t = torch.cat((self.memory_t, batch_norm_gt[:8]), dim=1) # past norm: [8, N, 2]
                self.memory_s = torch.cat((self.memory_s, x_social_), dim=1) # social encoding: [8, N, D]
                # self.memory_key = torch.cat((self.memory_key, batch_norm_gt[[x + 8 for x in self.args.key_points]]), dim=1) # key points: [4, N, 2]
                self.memory_key = torch.cat((self.memory_key, key_points), dim=1) # key points: [4, N, 2]
                # self.memory_y = torch.cat((self.memory_y, batch_norm_gt[8:]), dim=1) # future norm: [12, N, 2]
                self.shift_values = torch.cat((self.shift_values, shift_values), dim=1)
                self.max_values = torch.cat((self.max_values, max_values), dim=0)

                nbr_tmp = [] # 用于存储每个agent的完整邻居轨迹
                for n in range(batch_abs_gt.size(1)):
                    nbr_tmp.append([])
                start_idx = 0
                for i, (left, right) in enumerate(batch_split):
                    batch_abs_gt_now = batch_abs_gt[:, left:right]
                    edge_pair_now = edge_pair[i]
                    for f, t in edge_pair_now: # f与t是batch内部的index,需要转化为全局index
                        t_global = int(t) + start_idx
                        nbr_tmp[t_global].append(batch_abs_gt_now[:, f])
                    start_idx += int(right - left)
                
                self.complete_nbr.extend(nbr_tmp)
        
        # import matplotlib.pyplot as plt
        # n_max = self.memory_t.size(1)
        
        # for n in range(0, n_max, 200):
        #     memory_t_x = self.memory_t[:, n, 0].cpu().numpy()
        #     memory_t_y = self.memory_t[:, n, 1].cpu().numpy()
        #     memory_key_x = self.memory_key[:, n, 0].cpu().numpy()
        #     memory_key_y = self.memory_key[:, n, 1].cpu().numpy()
        #     memory_y_x = self.memory_y[:, n, 0].cpu().numpy()
        #     memory_y_y = self.memory_y[:, n, 1].cpu().numpy()

        #     # 绘制蓝色线条和点
        #     plt.plot(memory_t_x, memory_t_y, 'bo-')
        #     plt.quiver(memory_t_x[-2], memory_t_y[-2],
        #             memory_t_x[-1] - memory_t_x[-2], memory_t_y[-1] - memory_t_y[-2],
        #             angles='xy', scale_units='xy', color='blue', width = 0.005, scale = 1.03)
            
        #     # 绘制绿色线条和点
        #     plt.plot(memory_y_x, memory_y_y, 'go-')
        #     plt.quiver(memory_y_x[-2], memory_y_y[-2],
        #             memory_y_x[-1] - memory_y_x[-2], memory_y_y[-1] - memory_y_y[-2],
        #             angles='xy', scale_units='xy', color='green', width = 0.005, scale = 1.03)

        #     # 绘制红色线条和点
        #     plt.plot(memory_key_x, memory_key_y, 'ro-')
        #     plt.quiver(memory_key_x[-2], memory_key_y[-2],
        #             memory_key_x[-1] - memory_key_x[-2], memory_key_y[-1] - memory_key_y[-2],
        #             angles='xy', scale_units='xy', color='red', width = 0.005, scale = 1.03)
            

        #     # # 每隔 10 次显示一次图形
        #     # if n % 100 == 0 and n != 0:
        #     #     plt.show()
        #     #     plt.close()
        #     plt.gca().tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        #     plt.savefig("./bank_vis_avg/memory_%d.jpg" % n)
        #     plt.clf()
        # return
        
        if self.args.memory_filter:
            '''
            决定记忆删去的依据
            1、在历史轨迹相似的情况下, social encoding也相似
            
            如何 解耦 观测轨迹 和 social encoding ?
            问题: 对于某个social encoding, 我希望剔除原始的自身特征, 转化为适应当前输入的social encoding

            记忆库存储的: (past_traj, social encoding, key points)
            '''
            
            index = [0]
            t_s = 0.5
            t_t = 0.5
            t_k = 2
            temporal_memory = self.memory_t[:, 0:1] # choose the first memory as the start
            spatial_memory = self.memory_s[:, 0:1]
            key_memory = self.memory_key[:, 0:1]
            
            num_sample = self.memory_s.shape[1]
            for i in range(1, num_sample): # loop through all samples
                distances_t = torch.norm(temporal_memory - self.memory_t[:, [i]], p=2, dim=-1).mean(0) # past distance
                distances_s = torch.cosine_similarity(spatial_memory, self.memory_s[:, [i]], dim=-1).mean(0) # spatial consine similarity
                distances_key = torch.norm(key_memory - self.memory_key[:, [i]], p=2, dim=-1).mean(0) # key point distance
                mask_t = torch.where(distances_t < t_t, torch.ones_like(distances_t), torch.zeros_like(distances_t))
                mask_s = torch.where(distances_s < t_s, torch.ones_like(distances_s), torch.zeros_like(distances_s))
                mask_k = torch.where(distances_key < t_k, torch.ones_like(distances_key), torch.zeros_like(distances_key))
                mask = mask_s + mask_t + mask_k
                min_distance = torch.max(mask).item()
                if min_distance < 3: # omit this sample
                    index.append(i) # add the index
                    temporal_memory = torch.cat((temporal_memory, self.memory_t[:, [i]]), dim=1) # add the temporal to memory
                    spatial_memory = torch.cat((spatial_memory, self.memory_s[:, [i]]), dim=1) # add the spatial to memory
                    key_memory = torch.cat((key_memory, self.memory_key[:, [i]]), dim=1) # add the key point to memory
            self.memory_t = self.memory_t[:, index]
            self.memory_s = self.memory_s[:, index]
            self.memory_key = self.memory_key[:, index]
            # self.memory_y = self.memory_y[:, index]
            self.shift_values = self.shift_values[:, index]
            self.max_values = self.max_values[index]
            self.complete_nbr = [self.complete_nbr[i] for i in index]
        torch.save(self.memory_t, os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'memory_t_filtered_{}.pt'.format(epoch)))
        torch.save(self.memory_s, os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'memory_s_filtered_{}.pt'.format(epoch)))
        torch.save(self.memory_key, os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'memory_key_filtered_{}.pt'.format(epoch)))
        torch.save(self.complete_nbr, os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'complete_nbr_filtered_{}.pt'.format(epoch)))
        torch.save(self.shift_values, os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'shift_values_filtered_{}.pt'.format(epoch)))
        torch.save(self.max_values, os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'max_values_filtered_{}.pt'.format(epoch)))


    def load_memory(self, epoch):
        epoch = self.args.load_memory
        if epoch > 0 and self.args.train_phase != 'gen_memory':
            self.net.memory_t = torch.load(os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'memory_t_filtered_{}.pt'.format(epoch)))
            self.net.memory_s = torch.load(os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'memory_s_filtered_{}.pt'.format(epoch)))
            self.net.memory_key = torch.load(os.path.join(self.args.save_base_dir, str(self.args.test_set), 'MDTraj', 'memory_key_filtered_{}.pt'.format(epoch)))

    def test_epoch(self, epoch):
        self.net.eval()
        

        error_epoch, final_error_epoch = {}, {}
        error_cnt_epoch, final_error_cnt_epoch = {}, {}

        avg_ade, avg_fde = 0, 0
        avg_ade_cnt, avg_fde_cnt = 1e-5, 1e-5

        testloader_gt_progress = tqdm(enumerate(self.testloader_gt), total=len(self.testloader_gt), ncols=ncols)

        for batch, (batch_abs_gt, batch_norm_gt, key_points, shift_values, max_values, batch_split, edge_pair) in testloader_gt_progress:
            batch_abs_gt = batch_abs_gt[0].float().cuda()
            batch_norm_gt = batch_norm_gt[0].float().cuda()
            key_points = key_points[0].float().cuda()
            shift_values = shift_values[0].float().cuda()
            max_values = max_values[0].float().cuda()

            edge_pair = [x[0].long().cuda() for x in edge_pair]
            inputs_fw = batch_abs_gt, batch_norm_gt, key_points, shift_values, max_values, batch_split, edge_pair

            batch_class = batch_abs_gt[-1, :, -1].long()
            if self.args.test_set != 9: # 非sdd, 全部是行人
                batch_class = torch.zeros_like(batch_class)
            batch_class_unique = torch.unique(batch_class)

            full_pre_tra = self.net.forward(inputs_fw)

            full_pre_tra = [x * max_values for x in full_pre_tra]
            batch_norm_gt = batch_norm_gt * max_values

            for clss in batch_class_unique:
                clss = int(clss.item())
                mask = (batch_class == clss)
                # 计算mask这一类的metric
                error_epoch_min, final_error_epoch_min = [], []
                for pre_tra in full_pre_tra:
                    error, error_cnt, final_error, final_error_cnt = \
                    self.L2forTest(pre_tra[:, mask, ...], batch_norm_gt[1:, mask, :2], self.args.obs_length)

                    error_epoch_min.append(error)
                    final_error_epoch_min.append(final_error)

                error_epoch_min = min(error_epoch_min)
                final_error_epoch_min = min(final_error_epoch_min)

                error_epoch.setdefault(clss, 0)
                final_error_epoch.setdefault(clss, 0)

                error_cnt_epoch.setdefault(clss, 1e-5)
                final_error_cnt_epoch.setdefault(clss, 1e-5)

                final_error_epoch[clss] += final_error_epoch_min
                error_epoch[clss] += error_epoch_min

                error_cnt_epoch[clss] += error_cnt
                final_error_cnt_epoch[clss] += final_error_cnt

                avg_ade += error_epoch_min
                avg_fde += final_error_epoch_min
                avg_ade_cnt += error_cnt
                avg_fde_cnt += final_error_cnt

        for k in error_epoch.keys():
            error_epoch[k] /= error_cnt_epoch[k]
            final_error_epoch[k] /= final_error_cnt_epoch[k]

        error_epoch['avg'] = avg_ade / avg_ade_cnt
        final_error_epoch['avg'] = avg_fde / avg_fde_cnt

        return error_epoch, final_error_epoch
