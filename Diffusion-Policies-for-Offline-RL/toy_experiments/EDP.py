import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if 'Losses' not in globals():
    Losses = {
        'l1': nn.L1Loss(reduction='none'),
        'l2': nn.MSELoss(reduction='none'),
    }

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32): # Changed dtype
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=dtype) # dtype will be float32
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float() # Ensure float32

def vp_beta_schedule(timesteps):
    t = torch.arange(1, timesteps + 1, dtype=torch.float32) # Ensure float32
    T = float(timesteps) # Ensure T is float for division
    b_max = 10.0
    b_min = 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1.0 - alpha
    return betas.float() # Ensure float32

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class Progress:
    def __init__(self, total, name='Progress'):
        self.total = total
        self.name = name
        self.current = 0
        self.metrics = {}
    def update(self, metrics_dict=None):
        self.current += 1
        if metrics_dict: self.metrics.update(metrics_dict)
    def close(self): pass

class Silent:
    def update(self, *args, **kwargs): pass
    def close(self, *args, **kwargs): pass

class EDP(nn.Module):
    def __init__(self, state_dim, action_dim, epsilon_theta_network, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 time_embedding_dim=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(EDP, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon_theta_network = epsilon_theta_network.to(device).float()
        self.max_action = max_action
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.device = device
        self.time_embedding_dim = time_embedding_dim if time_embedding_dim is not None else action_dim * 4

        if beta_schedule == 'linear': betas = linear_beta_schedule(self.n_timesteps)
        elif beta_schedule == 'cosine': betas = cosine_beta_schedule(self.n_timesteps, dtype=torch.float32)
        elif beta_schedule == 'vp': betas = vp_beta_schedule(self.n_timesteps)
        else: raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        betas = betas.to(device).float()
        alphas = (1. - betas).float()
        alphas_cumprod = torch.cumprod(alphas, axis=0).float()
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device, dtype=torch.float32), alphas_cumprod[:-1]]).float()

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).float())
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())

        posterior_variance = (betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)).float()
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)).float())
        self.register_buffer('posterior_mean_coef1', (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).float())
        self.register_buffer('posterior_mean_coef2', ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)).float())

        if isinstance(loss_type, str) and loss_type in Losses:
            self.loss_fn = Losses[loss_type]
        elif callable(loss_type):
            self.loss_fn = loss_type
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        if self.time_embedding_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_embedding_dim // 2, self.time_embedding_dim),
                nn.Mish(),
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            ).to(device).float()
        else:
            self.time_mlp = None

    def _time_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=self.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.float()

    def _predict_noise_from_model(self, a_k, t_int, state):
        state = state.float()
        a_k = a_k.float()
        if self.time_embedding_dim is not None:
            if not isinstance(t_int, torch.Tensor): t_int = torch.tensor([t_int], device=self.device, dtype=torch.long)
            elif t_int.ndim == 0: t_int = t_int.unsqueeze(0)
            if t_int.dtype != torch.long: t_int = t_int.long()
            t_int_device = t_int.to(self.device)
            t_emb = self._time_embedding(t_int_device, self.time_embedding_dim // 2)
            t_emb = self.time_mlp(t_emb).float()
            if t_emb.shape[0] < a_k.shape[0] and a_k.shape[0] % t_emb.shape[0] == 0:
                repeats = a_k.shape[0] // t_emb.shape[0]
                t_emb = t_emb.repeat_interleave(repeats, dim=0)
                state_for_model = state.repeat_interleave(repeats, dim=0)
            else:
                state_for_model = state
            return self.epsilon_theta_network(a_k, t_emb, state_for_model).float()
        else:
            return self.epsilon_theta_network(a_k, state=state).float()

    def predict_start_from_noise(self, a_k, k_int_timestep, predicted_noise_epsilon):
        a_k = a_k.float()
        predicted_noise_epsilon = predicted_noise_epsilon.float()
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, k_int_timestep, a_k.shape) * a_k -
                    extract(self.sqrt_recipm1_alphas_cumprod, k_int_timestep, a_k.shape) * predicted_noise_epsilon
            ).float()
        else: return predicted_noise_epsilon.float()

    def action_approximation(self, state, action_dataset_a0, k_int_timestep_or_tensor):
        action_dataset_a0 = action_dataset_a0.float()
        state = state.float()
        if isinstance(k_int_timestep_or_tensor, int): t_int = torch.full((action_dataset_a0.shape[0],), k_int_timestep_or_tensor, device=self.device, dtype=torch.long)
        else: t_int = k_int_timestep_or_tensor.to(self.device).long()

        true_noise_epsilon = torch.randn_like(action_dataset_a0).float()
        a_k_corrupted = self.q_sample(x_start=action_dataset_a0, t_int_timesteps=t_int, noise=true_noise_epsilon).float()
        predicted_noise_epsilon = self._predict_noise_from_model(a_k_corrupted, t_int, state).float()
        a_hat_0_approximated = self.predict_start_from_noise(a_k_corrupted, t_int, predicted_noise_epsilon).float()

        if self.clip_denoised: a_hat_0_approximated = torch.clamp(a_hat_0_approximated, -self.max_action, self.max_action)
        return a_hat_0_approximated, a_k_corrupted, predicted_noise_epsilon

    def dpm_solver_sample_placeholder(self, state, verbose=False, num_steps=15):
        return self.p_sample_loop(state, shape=(state.shape[0], self.action_dim), verbose=verbose, use_dpm_solver_pseudo_steps=num_steps)

    def eas_sample(self, state, q_function, num_eas_samples=10, sampler_type='dpm_solver'):
        batch_size = state.shape[0]
        candidate_actions_list = []
        for _ in range(num_eas_samples):
            if sampler_type == 'dpm_solver': actions_i = self.dpm_solver_sample_placeholder(state.float(), num_steps=15)
            else: actions_i = self.p_sample_loop(state.float(), shape=(batch_size, self.action_dim))
            candidate_actions_list.append(actions_i.unsqueeze(1))

        candidate_actions_tensor = torch.cat(candidate_actions_list, dim=1)
        flat_candidate_actions = candidate_actions_tensor.reshape(batch_size * num_eas_samples, self.action_dim).float()
        expanded_states = state.unsqueeze(1).repeat(1, num_eas_samples, 1).reshape(batch_size * num_eas_samples, self.state_dim).float()

        with torch.no_grad(): q_values = q_function(expanded_states, flat_candidate_actions).reshape(batch_size, num_eas_samples).float()
        q_values_stable = q_values - torch.max(q_values, dim=1, keepdim=True)[0]
        probabilities = F.softmax(q_values_stable, dim=1)
        action_indices = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        final_actions = candidate_actions_tensor[torch.arange(batch_size, device=self.device), action_indices]
        return final_actions.clamp_(-self.max_action, self.max_action).float()

    def q_sample(self, x_start, t_int_timesteps, noise=None):
        x_start = x_start.float()
        if noise is None: noise = torch.randn_like(x_start).float()
        else: noise = noise.float()
        return (extract(self.sqrt_alphas_cumprod, t_int_timesteps, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t_int_timesteps, x_start.shape) * noise).float()

    def p_mean_variance(self, a_k_corrupted, k_int_timestep, s_state, grad_through_model=False):
        a_k_corrupted = a_k_corrupted.float()
        s_state = s_state.float()
        if grad_through_model: model_output_epsilon = self._predict_noise_from_model(a_k_corrupted, k_int_timestep, s_state)
        else:
            with torch.no_grad(): model_output_epsilon = self._predict_noise_from_model(a_k_corrupted, k_int_timestep, s_state)
        model_output_epsilon = model_output_epsilon.float()
        a_0_reconstructed = self.predict_start_from_noise(a_k_corrupted, k_int_timestep, model_output_epsilon).float()
        if self.clip_denoised: a_0_reconstructed.clamp_(-self.max_action, self.max_action)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=a_0_reconstructed, x_t=a_k_corrupted, t_int_timestep=k_int_timestep)
        return model_mean.float(), posterior_variance.float(), posterior_log_variance.float()

    def q_posterior(self, x_start, x_t, t_int_timestep):
        x_start = x_start.float()
        x_t = x_t.float()
        posterior_mean = (extract(self.posterior_mean_coef1, t_int_timestep, x_t.shape) * x_start +
                          extract(self.posterior_mean_coef2, t_int_timestep, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t_int_timestep, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t_int_timestep, x_t.shape)
        return posterior_mean.float(), posterior_variance.float(), posterior_log_variance_clipped.float()

    def p_sample(self, a_k_corrupted, k_int_timestep, s_state, grad_through_model=False):
        a_k_corrupted = a_k_corrupted.float()
        s_state = s_state.float()
        batch_size = a_k_corrupted.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(a_k_corrupted, k_int_timestep, s_state, grad_through_model)
        noise = torch.randn_like(a_k_corrupted).float()
        nonzero_mask = (1 - (k_int_timestep.to(self.device) == 0).float()).reshape(batch_size, *((1,) * (len(a_k_corrupted.shape) - 1)))
        return (model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise).float()

    def p_sample_loop(self, state_condition, shape, verbose=False, return_diffusion=False, use_dpm_solver_pseudo_steps=None):
        state_condition = state_condition.float()
        batch_size = shape[0]
        a_k_current = torch.randn(shape, device=self.device).float()
        if return_diffusion: diffusion_history = [a_k_current.cpu()]

        num_loop_steps = use_dpm_solver_pseudo_steps if use_dpm_solver_pseudo_steps is not None else self.n_timesteps
        progress_bar = Progress(num_loop_steps, name="DDPM Sampling") if verbose else Silent()

        for i in reversed(range(0, num_loop_steps)):
            current_k_int_val = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            a_k_current = self.p_sample(a_k_current, current_k_int_val, state_condition, grad_through_model=False)
            progress_bar.update({'t': i})
            if return_diffusion: diffusion_history.append(a_k_current.cpu())
        progress_bar.close()

        final_action = a_k_current.clamp_(-self.max_action, self.max_action)
        if return_diffusion: return final_action.float(), torch.stack(diffusion_history, dim=1).float()
        return final_action.float()

    def diffusion_loss(self, action_dataset_a0, state_condition, k_int_timestep_tensor=None):
        action_dataset_a0 = action_dataset_a0.float()
        state_condition = state_condition.float()
        batch_size = action_dataset_a0.shape[0]
        if k_int_timestep_tensor is None: t_int = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        else: t_int = k_int_timestep_tensor.to(self.device).long()

        true_noise_epsilon = torch.randn_like(action_dataset_a0).float()
        a_k_corrupted = self.q_sample(x_start=action_dataset_a0, t_int_timesteps=t_int, noise=true_noise_epsilon).float()
        model_output_epsilon = self._predict_noise_from_model(a_k_corrupted, t_int, state_condition).float()

        if self.predict_epsilon: loss = self.loss_fn(model_output_epsilon, true_noise_epsilon)
        else: loss = self.loss_fn(model_output_epsilon, action_dataset_a0)
        return loss.mean()

    def edp_likelihood_based_policy_loss(self, state_batch, action_batch_a0, q_function, adv_weights_f_Q):
        state_batch = state_batch.float()
        action_batch_a0 = action_batch_a0.float()
        adv_weights_f_Q = adv_weights_f_Q.float()
        batch_size = state_batch.shape[0]
        k_int_timesteps_uniform = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        a_hat_0_approximated, _, _ = self.action_approximation(state_batch, action_batch_a0, k_int_timesteps_uniform)
        diff_sq_loss = self.loss_fn(action_batch_a0, a_hat_0_approximated.float())
        weighted_loss = adv_weights_f_Q * diff_sq_loss
        return weighted_loss.mean()

    def forward(self, state_condition, q_function_for_eas=None, eas_num_samples=10, evaluation_sampler_type='dpm_solver'):
        self.epsilon_theta_network.eval()
        state_condition = state_condition.float()
        if q_function_for_eas is not None and eas_num_samples > 0:
            if callable(q_function_for_eas): q_function_for_eas.eval()
            return self.eas_sample(state_condition, q_function_for_eas, num_eas_samples=eas_num_samples, sampler_type=evaluation_sampler_type)
        elif evaluation_sampler_type == 'dpm_solver':
            return self.dpm_solver_sample_placeholder(state_condition, num_steps=15)
        else:
            return self.p_sample_loop(state_condition, shape=(state_condition.shape[0], self.action_dim))