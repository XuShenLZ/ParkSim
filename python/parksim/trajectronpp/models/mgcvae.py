from parksim.trajectronpp.models.common_blocks import BaseMGCVAELightningModule
from parksim.trajectronpp.models.components import GMM2D
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td

class MGCVAE(nn.Module):
    # TODO: fill this in
    # TODO: account for prediction mode too
    # TODO: figure out what hyperparams this takes
    def __init__(self, config) -> None:
        super().__init__()

        # INFO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # TODO: update these for accuracy
        self.model_type = "Multimodal Generative CVAE"
        self.dim_model = 2
        # TODO: make sure i understand what N and K are
        self.N = config['N']
        self.K = config['K']
        # TODO: this number is totally fake, gotta figure out how to do the annealing situation with lightning (maybe?) or just manually
        self.kl_weight = 0.1
        # distributions for loss
        self.q_dist = None
        self.p_dist = None

        # LAYERS
        # TODO: update w layers from paper & sizes from data/config
        # TODO: figure out how to retreive from saved locations
        self.state_length = 4
        self.pred_state_length = 4
        self.pred_horizon = config["prediction_horizon"]
        # Encoder Layers
        self.target_history_encoder = nn.LSTM(input_size=self.state_length, hidden_size=32, batch_first=True)
        self.neighbor_veh_history_encoder = nn.LSTM(input_size=self.state_length, hidden_size=32, batch_first=True)
        self.neighbor_ped_history_encoder = nn.LSTM(input_size=self.state_length, hidden_size=32, batch_first=True)
        self.node_future_encoder = nn.LSTM(input_size=self.state_length, hidden_size=32, batch_first=True)
        self.node_future_h0 = nn.Linear(self.state_length, 32)
        self.node_future_c0 = nn.Linear(self.state_length, 32)
        self.x_size = 128
        self.z_dim = self.N * self.K
        # Discrete Latent Variable
        self.p_z_x = nn.Linear(self.x_size, 8)
        self.hx_to_z = nn.Linear(8, self.z_dim)
        self.q_z_xy = nn.Linear(self.x_size + 32, 8)
        self.hxy_to_z = nn.Linear(8, self.z_dim)
        # Decoder LSTM Layers
        self.state_action = nn.Linear(self.state_length, self.pred_state_length)
        self.rnn_cell = nn.GRUCell(self.pred_state_length+self.z_dim+self.x_size, config['dec_rnn_dim'])
        self.initial_h = nn.Linear(self.z_dim+self.x_size, config['dec_rnn_dim'])
        # Decoder GMM
        self.proj_to_GMM_log_pis = nn.Linear(config['dec_rnn_dim'], config['GMM_components'])
        self.proj_to_GMM_mus = nn.Linear(config['dec_rnn_dim'], config['GMM_components']*self.pred_state_length)
        self.proj_to_GMM_log_sigmas = nn.Linear(config['dec_rnn_dim'], config['GMM_components']*self.pred_state_length)
        self.proj_to_GMM_corrs = nn.Linear(config['dec_rnn_dim'], config['GMM_components'])
        # TODO: update w correct channel numbers and dense layer dims
        self.map_encoder = nn.Sequential(nn.Conv2d(3, 3, 5, 2), nn.Conv2d(3, 3, 5, 2), nn.Conv2d(3, 3, 5, 1), nn.Conv2d(3, 3, 3, 1), nn.Flatten(), nn.Linear(24843, 32))
        # self.layer2 = nn.Linear(20, 3)

    @staticmethod
    def all_one_hot_combinations(N, K):
        return np.eye(K).take(np.reshape(np.indices([K] * N), [N, -1]).T, axis=0).reshape(-1, N * K)  # [K**N, N*K]
    
    @staticmethod
    def mutual_inf_mc(x_dist):
        dist = x_dist.__class__
        H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
        return (H_y - x_dist.entropy().mean(dim=0)).sum()
    
    def dist_from_h(self, h, mode):
        logits_separated = torch.reshape(h, (-1, self.N, self.K))
        logits = logits_separated - torch.mean(logits_separated, dim=-1, keepdim=True)
        return td.OneHotCategorical(logits=logits)

    def encode_tensors(self, mode, target_history, neighbor_veh_history, neighbor_ped_history, node_future, map):
        # TODO: account for mode
        target_history_enc, _ = self.target_history_encoder(target_history)
        neighbor_ped_enc, _ = self.neighbor_ped_history_encoder(neighbor_ped_history)
        neighbor_veh_enc, _ = self.neighbor_veh_history_encoder(neighbor_veh_history)
        map_enc = self.map_encoder(map).unsqueeze(1).repeat(1, 10, 1)
        x_concat_list = [target_history_enc, neighbor_ped_enc, neighbor_veh_enc, map_enc]
        # print(f"{target_history_enc.shape=}")
        # print(f"{neighbor_ped_enc.shape=}")
        # print(f"{neighbor_veh_enc.shape=}")
        # print(f"{map_enc.shape=}")
        x = torch.cat(x_concat_list, dim=2)
        # TODO: node_future_c0? node_future_h0?
        y, _ = self.node_future_encoder(node_future)
        # TODO: normalize n_s_t0?
        n_s_t0 = target_history[:,-1]
        return x, y, n_s_t0
    
    def p_y_xz(self, mode, x, n_s_t0, z_stacked, prediction_horizon, num_samples, num_components=1):
        z = torch.reshape(z_stacked, (-1, self.z_dim))
        # print(f"{z.shape=}")
        # print(f"{x.shape=}")
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)
        initial_state = self.initial_h(zx)
        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []
        a_0 = self.state_action(n_s_t0)
        state = initial_state
        input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(prediction_horizon):
            h_state = self.rnn_cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            a_t = gmm.rsample()

            if num_components > 1:
                log_pis.append(self.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, num_components))

            dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, prediction_horizon, num_components]),
                       torch.reshape(mus, [num_samples, -1, prediction_horizon, num_components * self.pred_state_length]),
                       torch.reshape(log_sigmas, [num_samples, -1, prediction_horizon, num_components * self.pred_state_length]),
                       torch.reshape(corrs, [num_samples, -1, prediction_horizon, num_components]))
        
        y_dist = a_dist
        return y_dist

    def encoder(self, mode, x, y):
        # TODO: account for mode
        xy = torch.cat((x,y), dim=2)
        # print(f"{xy.shape=}")
        q = self.q_z_xy(xy)
        hxy = self.hxy_to_z(q)
        self.q_dist = self.dist_from_h(hxy, mode)
        self.p_dist = self.dist_from_h(self.hx_to_z(self.p_z_x(x)), mode)
        # TODO: num samples is mode dependent?
        num_samples = 1
        bs = self.p_dist.probs.size()[0]
        num_components = self.N * self.K
        z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(num_samples, bs)
        z = torch.reshape(z_NK, (num_samples * num_components, -1, self.z_dim))
        kl_separated = td.kl_divergence(self.q_dist, self.p_dist)
        kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)
        kl_obj = torch.sum(kl_minibatch)
        return z, kl_obj

    def decoder(self, mode, x, n_s_t0, z, labels, prediction_horizon, num_samples):
        num_components = self.N*self.K
        y_dist = self.p_y_xz(mode, x, n_s_t0, z, prediction_horizon, num_samples, num_components)
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        return log_p_y_xz
    
    def get_training_loss(self, log_p_y_xz, kl):
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = MGCVAE.mutual_inf_mc(self.q_dist)
        mutual_inf_p = MGCVAE.mutual_inf_mc(self.p_dist)

        ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        loss = -ELBO
        return loss

    def  forward(self, mode, target_history, neighbor_veh_history, neighbor_ped_history, node_future, map):
        # middle = self.layer1(target_history)
        # out, _ = self.target_history_encoder(target_history)
        if mode == "train":
            x, y, n_s_t0 = self.encode_tensors(mode, target_history, neighbor_veh_history, neighbor_ped_history, node_future, map)
            z, kl_obj = self.encoder(mode, x, y)
            log_p_y_xz = self.decoder(mode, x, n_s_t0, z, node_future, self.pred_horizon, self.K)
            loss = self.get_training_loss(log_p_y_xz, kl_obj)
        return loss

# TODO: fill this in
DEFAULT_CONFIG = {}

class TrajectoryPredictorMGCVAE(BaseMGCVAELightningModule):
    # TODO: fill this in
    def __init__(self, config: dict, input_shape=(), loss_fn=F.l1_loss) -> None:
        super().__init__(config, input_shape, loss_fn)
        # TODO: include the other models & update this one
        self.model = MGCVAE(config)
    
    def forward(self, ego_history, target_history, neighbor_veh_history, neighbor_ped_history, target_future, semantic_map):
        # TODO: update this to be accurate
        # TODO: do i need to take the target future?
        """
        ego_history:            (N, T_1, 3)
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)
                                
        target_history:         (N, T_1, 3)     
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)
                                
        neighbor_veh_history:   (N, T_1, 3)     
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)

        neighbor_ped_history:   (N, T_1, 3)     
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)

        semantic_map:           (N, 3, 100, 100)     
                                N = batch size
                                Image corresponding to instance centric view at current time step


        Returns - 

        output:                 (N, T_2, 3)
                                N = batch size
                                T_2 = timesteps of output
                                3 = (x_coord, y_coord, heading)

        """
        # TODO: really need to update this one for eval and test
        # TODO: node future?
        output = self.model("train", target_history, neighbor_veh_history, neighbor_ped_history, target_future, semantic_map)
        return output
