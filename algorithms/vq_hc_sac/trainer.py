"""
VQ-HC-SAC Trainer: End-to-end training loop.

Integrates all components for complete VQ-HC-SAC training:
- Phase 1: Networks (Encoder, VQ, Actor, Critic)
- Phase 2: Components (RoleManager, ReplayBuffer, TemperatureManager)
- Phase 3: Losses (SAC, VQ)
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from collections import deque

from .config import VQHCSACConfig
from .networks import StateEncoder, VQRoleModule
from .components import RoleManager, MultiAgentReplayBuffer, TemperatureManager
from .losses import SACLoss, VQLoss, VQStatistics


class VQHCSACTrainer:
    """
    Main trainer for VQ-HC-SAC algorithm.
    
    Training flow:
    1. Collect experience from environment
    2. Sample batch from replay buffer
    3. Assign roles via VQ module
    4. Compute role aggregations
    5. Update critics (M role-specific)
    6. Update actors (M role-specific)
    7. Update alphas (M role-specific)
    8. Update VQ encoder and codebook
    """
    
    def __init__(
        self,
        env,
        config: VQHCSACConfig,
        save_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize VQ-HC-SAC trainer.
        
        Args:
            env: Multi-agent environment (Gymnasium interface)
            config: Training configuration
            save_dir: Directory for saving checkpoints
            device: Device for training ('cuda' or 'cpu')
        """
        self.env = env
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path('./checkpoints')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Extract environment dimensions
        self._extract_env_dimensions()
        
        # Initialize networks
        self._init_networks()
        
        # Initialize components
        self._init_components()
        
        # Initialize losses
        self._init_losses()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.best_eval_return = -float('inf')
        
        # Statistics tracking
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Debug flags
        self._first_critic_update = True
        
        print(f"VQ-HC-SAC Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Roles: {config.n_roles}")
        print(f"  Agents: {self.n_agents}")
        print(f"  Save dir: {self.save_dir}")
        
    def _ts(self):
        t = time.time()
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
    
    def _extract_env_dimensions(self):
        """Extract dimensions from environment."""
        # Get sample observation
        obs, _ = self.env.reset()
        
        # Handle different observation formats
        if isinstance(obs, tuple):
            # Multi-agent tuple of observations
            self.n_agents = len(obs)
            individual_obs = obs[0]
        else:
            # Single observation (will need to adapt)
            self.n_agents = 1
            individual_obs = obs
        
        # Get dimensions
        if hasattr(individual_obs, 'shape'):
            self.individual_state_dim = individual_obs.shape[-1]
        else:
            self.individual_state_dim = len(individual_obs)
        
        # Get action dimension
        # Handle Tuple action space (multi-agent)
        if hasattr(self.env.action_space, '__iter__') and not hasattr(self.env.action_space, 'n'):
            # Tuple of action spaces
            first_action_space = self.env.action_space[0]
            if hasattr(first_action_space, 'n'):
                self.action_dim = first_action_space.n
                self.action_type = 'discrete'
            else:
                self.action_dim = first_action_space.shape[0]
                self.action_type = 'continuous'
        elif hasattr(self.env.action_space, 'n'):
            self.action_dim = self.env.action_space.n
            self.action_type = 'discrete'
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.action_type = 'continuous'
        
        # Environment state dimension (simplified: sum of all obs)
        self.env_state_dim = self.individual_state_dim * self.n_agents
        
        # Environment dimensions extracted
    
    def _init_networks(self):
        """Initialize neural networks."""
        # State encoder
        self.encoder = StateEncoder(
            input_dim=self.individual_state_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.encoder_hidden_dims
        ).to(self.device)
        
        # VQ role module
        self.vq_module = VQRoleModule(
            n_roles=self.config.n_roles,
            embedding_dim=self.config.embedding_dim,
            encoder=self.encoder,
            commitment_cost=self.config.vq_beta,
            use_ema=self.config.use_ema_codebook,
            ema_decay=self.config.codebook_ema_decay
        ).to(self.device)
        
        # Networks initialized
    
    def _init_components(self):
        """Initialize training components."""
        # Role manager
        self.role_manager = RoleManager(
            n_roles=self.config.n_roles,
            individual_state_dim=self.individual_state_dim,
            action_dim=self.action_dim,
            env_state_dim=self.env_state_dim,
            role_aggregation_dim=self.config.embedding_dim,
            actor_hidden_dims=self.config.actor_hidden_dims,
            critic_hidden_dims=self.config.critic_hidden_dims,
            action_type=self.action_type,
            use_twin_critic=self.config.use_double_critic,
            device=self.device
        )
        
        # Replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(
            capacity=self.config.buffer_size,
            n_agents=self.n_agents,
            individual_state_dim=self.individual_state_dim,
            env_state_dim=self.env_state_dim,
            action_dim=self.action_dim,
            action_type=self.action_type,
            device=self.device
        )
        
        # Temperature manager
        self.temp_manager = TemperatureManager(
            n_roles=self.config.n_roles,
            action_dim=self.action_dim,
            target_entropy_scale=self.config.target_entropy_scale,
            initial_alpha=self.config.initial_alpha,
            lr=self.config.alpha_lr,
            device=self.device
        )
        
        # Components initialized
    
    def _init_losses(self):
        """Initialize loss calculators."""
        self.sac_loss = SACLoss(
            gamma=self.config.gamma,
            use_twin_critic=self.config.use_double_critic
        )
        
        self.vq_loss = VQLoss(
            commitment_cost=self.config.vq_beta,
            use_ema_codebook=self.config.use_ema_codebook
        )
        
        # Losses initialized
    
    def _init_optimizers(self):
        """Initialize optimizers."""
        # Actor optimizer (all M actors)
        self.actor_optimizer = optim.Adam(
            self.role_manager.get_actor_parameters(),
            lr=self.config.actor_lr
        )
        
        # Critic optimizer (all M critics)
        self.critic_optimizer = optim.Adam(
            self.role_manager.get_critic_parameters(),
            lr=self.config.critic_lr
        )
        
        # VQ optimizer (encoder + codebook)
        vq_params = list(self.encoder.parameters()) + list(self.vq_module.codebook.parameters())
        self.vq_optimizer = optim.Adam(vq_params, lr=self.config.encoder_lr)
        
        # Alpha optimizer is handled by TemperatureManager
        
        # Optimizers initialized
    
    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        log_interval: int = 1000
    ) -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            total_steps: Total training steps
            eval_interval: Steps between evaluations
            log_interval: Steps between logging
        
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*70}")
        print(f"Starting VQ-HC-SAC Training")
        print(f"{'='*70}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")
        print(f"Batch size: {self.config.batch_size}")
        
        history = {
            'episode_returns': [],
            'episode_lengths': [],
            'eval_returns': [],
            'losses': []
        }
        
        # Reset environment
        obs, _ = self.env.reset()
        episode_return = np.zeros(self.n_agents)
        episode_length = 0
        
        start_time = time.time()
        last_print_time = time.time()
        
        # Training loop
        while self.total_steps < total_steps:
            # Collect experience
            obs, episode_return, episode_length, episode_done = self._collect_step(
                obs, episode_return, episode_length
            )
            
            # Update networks (after warmup)
            if self.total_steps > self.config.warmup_steps:
                if self.total_steps % self.config.update_frequency == 0:
                    for update_idx in range(self.config.updates_per_step):
                        losses = self._update_networks()
                        
                        if losses is not None:
                            history['losses'].append(losses)
            
            # Episode end
            if episode_done:
                self.episode_count += 1
                self.episode_returns.append(episode_return.sum())
                self.episode_lengths.append(episode_length)
                
                history['episode_returns'].append(episode_return.sum())
                history['episode_lengths'].append(episode_length)
                
                # Reset for new episode
                obs, _ = self.env.reset()
                episode_return = np.zeros(self.n_agents)
                episode_length = 0
            
            # Logging
            if self.total_steps % log_interval == 0:
                self._log_training_progress(start_time)
            
            # Evaluation
            if self.total_steps % eval_interval == 0:
                eval_return = self.evaluate(n_episodes=self.config.eval_episodes)
                history['eval_returns'].append(eval_return)
                
                # Save best model
                if eval_return > self.best_eval_return:
                    self.best_eval_return = eval_return
                    self.save_checkpoint('best_model.pt')
            
            # Save checkpoint
            if self.total_steps % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_{self.total_steps}.pt')
            
            self.total_steps += 1
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Total episodes: {self.episode_count}")
        print(f"Total steps: {self.total_steps}")
        print(f"Best eval return: {self.best_eval_return:.2f}")
        
        return history
    
    def _collect_step(
        self,
        obs: Any,
        episode_return: np.ndarray,
        episode_length: int
    ) -> Tuple[Any, np.ndarray, int, bool]:
        """
        Collect one step of experience.
        
        Returns:
            next_obs, episode_return, episode_length, done
        """
        # Convert observations to tensors
        individual_states = self._process_observations(obs)
        env_state = self._compute_env_state(individual_states)
        
        # Assign roles and get actions
        actions, role_assignments = self._select_actions(individual_states)
        
        # Step environment
        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
        done = terminated or truncated
        
        # Process next state
        next_individual_states = self._process_observations(next_obs)
        next_env_state = self._compute_env_state(next_individual_states)
        
        # Convert rewards from dict to array
        if isinstance(rewards, dict):
            rewards_array = np.array([rewards[i] for i in range(self.n_agents)])
        else:
            rewards_array = np.array(rewards)
        
        # Get shaped rewards from info (if environment provides them)
        if 'shaped_rewards' in info:
            shaped_rewards = info['shaped_rewards']
            # Convert dict to array if needed
            if isinstance(shaped_rewards, dict):
                shaped_rewards = np.array([shaped_rewards[i] for i in range(self.n_agents)])
            else:
                shaped_rewards = np.array(shaped_rewards)
        else:
            # Use original rewards
            shaped_rewards = rewards_array
        
        # Store transition
        self.replay_buffer.add(
            individual_states.cpu().numpy(),
            env_state.cpu().numpy(),
            actions,
            shaped_rewards,
            next_individual_states.cpu().numpy(),
            next_env_state.cpu().numpy(),
            done,
            role_assignments.cpu().numpy()
        )
        
        # Update episode stats
        episode_return += rewards_array
        episode_length += 1
        
        return next_obs, episode_return, episode_length, done
    
    def _select_actions(
        self,
        individual_states: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Select actions for all agents.
        
        Returns:
            actions, role_assignments
        """
        with torch.no_grad():
            # Assign roles
            role_assignments = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)
            
            for i in range(self.n_agents):
                role_id = self.vq_module.get_role_assignment(individual_states[i])
                role_assignments[i] = role_id
            
            # Sample actions
            actions = []
            for i in range(self.n_agents):
                role_id = role_assignments[i].item()
                actor = self.role_manager.get_actor(role_id)
                
                # Deterministic during warmup, stochastic after
                deterministic = self.total_steps < self.config.warmup_steps
                action, log_prob = actor.sample(individual_states[i], deterministic=deterministic)
                
                if self.action_type == 'discrete':
                    actions.append(int(action.item()))  # 确保是Python int
                else:
                    actions.append(action.cpu().numpy())
            
            # 对于离散动作，保持为列表；连续动作转为array
            if self.action_type == 'continuous':
                actions = np.array(actions)
        
        return actions, role_assignments
    
    def _update_networks(self) -> Optional[Dict[str, float]]:
        """
        Update all networks.
        
        Returns:
            Dictionary of losses (or None if buffer not ready)
        """
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Update critics, actors, alphas, and VQ
        losses = {}
        losses.update(self._update_critics(batch))
        losses.update(self._update_actors(batch))
        losses.update(self._update_alphas(batch))
        losses.update(self._update_vq_module(batch))
        
        # Soft update target networks
        self.role_manager.soft_update_target_networks(self.config.tau)
        
        return losses
    
    def _update_critics(self, batch: Dict) -> Dict[str, float]:
        """
        Update critic networks for all roles. Vectorized version.
        """
        losses = {}
        
        batch_size = batch['individual_states'].shape[0]
        n_agents = batch['individual_states'].shape[1]
        
        # Reshape to [batch*agents, ...]
        flat_states = batch['individual_states'].view(batch_size * n_agents, -1)
        flat_next_states = batch['next_individual_states'].view(batch_size * n_agents, -1)
        flat_actions = batch['actions'].view(batch_size * n_agents, -1)
        flat_rewards = batch['shaped_rewards'].view(batch_size * n_agents)
        flat_role_assignments = batch['role_assignments'].view(batch_size * n_agents)
        
        # Expand dones and env_states
        flat_dones = batch['dones'].unsqueeze(1).expand(batch_size, n_agents).reshape(batch_size * n_agents)
        flat_env_states = batch['env_states'].unsqueeze(1).expand(batch_size, n_agents, -1).reshape(batch_size * n_agents, -1)
        flat_next_env_states = batch['next_env_states'].unsqueeze(1).expand(batch_size, n_agents, -1).reshape(batch_size * n_agents, -1)
        
        # Encode all states at once
        with torch.no_grad():
            batch_agent_embeddings = self.encoder(batch['individual_states'].view(-1, self.individual_state_dim)).view(batch_size, n_agents, -1)
            batch_next_agent_embeddings = self.encoder(batch['next_individual_states'].view(-1, self.individual_state_dim)).view(batch_size, n_agents, -1)
        
        # Compute role aggregations for all batches
        batch_role_agg = torch.stack([
            self.role_manager.compute_role_aggregations(batch_agent_embeddings[b], batch['role_assignments'][b], method='mean')
            for b in range(batch_size)
        ])
        batch_next_role_agg = torch.stack([
            self.role_manager.compute_role_aggregations(batch_next_agent_embeddings[b], batch['role_assignments'][b], method='mean')
            for b in range(batch_size)
        ])
        
        # Expand role aggregations
        flat_role_agg = batch_role_agg.unsqueeze(1).expand(batch_size, n_agents, self.config.n_roles, -1).reshape(batch_size * n_agents, self.config.n_roles, -1)
        flat_next_role_agg = batch_next_role_agg.unsqueeze(1).expand(batch_size, n_agents, self.config.n_roles, -1).reshape(batch_size * n_agents, self.config.n_roles, -1)
        
        # Process by role
        total_critic_loss = 0.0
        for role_id in range(self.config.n_roles):
            role_mask = (flat_role_assignments == role_id)
            if not role_mask.any():
                continue
            
            # Extract role-specific data
            role_states = flat_states[role_mask]
            role_actions = flat_actions[role_mask]
            role_rewards = flat_rewards[role_mask]
            role_next_states = flat_next_states[role_mask]
            role_dones = flat_dones[role_mask]
            role_env_states = flat_env_states[role_mask]
            role_next_env_states = flat_next_env_states[role_mask]
            role_agg_data = flat_role_agg[role_mask]
            role_next_agg_data = flat_next_role_agg[role_mask]
            
            # Get networks 
            critic = self.role_manager.get_critic(role_id)
            target_critic = self.role_manager.get_target_critic(role_id)
            actor = self.role_manager.get_actor(role_id)
            alpha = self.temp_manager.get_alpha(role_id)
            
            # Compute target Q values (vectorized)
            with torch.no_grad():
                next_actions, next_log_probs = actor.sample(role_next_states, deterministic=False)
                next_q1, next_q2 = target_critic(role_next_states, next_actions, role_next_env_states, role_next_agg_data)
                next_q = torch.min(next_q1, next_q2).squeeze(-1)
                target_q = role_rewards + (1 - role_dones) * self.config.gamma * (next_q - alpha * next_log_probs)
            
            # Current Q values
            q1, q2 = critic(role_states, role_actions, role_env_states, role_agg_data)
            
            # TD losses
            q1_loss = torch.nn.functional.mse_loss(q1.squeeze(-1), target_q)
            q2_loss = torch.nn.functional.mse_loss(q2.squeeze(-1), target_q)
            role_total_loss = q1_loss + q2_loss
            
            total_critic_loss += role_total_loss
            losses[f'critic_role_{role_id}_q1'] = q1_loss.item()
            losses[f'critic_role_{role_id}_q2'] = q2_loss.item()
        
        # Backpropagation
        if total_critic_loss > 0:
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            
            if self.config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.role_manager.get_critic_parameters(), self.config.clip_grad_norm)
            
            self.critic_optimizer.step()
            losses['critic_total'] = total_critic_loss.item()
        
        # Mark that first update is complete
        if self._first_critic_update:
            self._first_critic_update = False
        
        return losses
    
    def _update_actors(self, batch: Dict) -> Dict[str, float]:
        """
        Update actor networks for all roles. Vectorized version.
        """
        losses = {}
        
        batch_size = batch['individual_states'].shape[0]
        n_agents = batch['individual_states'].shape[1]
        
        # Reshape
        flat_states = batch['individual_states'].view(batch_size * n_agents, -1)
        flat_role_assignments = batch['role_assignments'].view(batch_size * n_agents)
        flat_env_states = batch['env_states'].unsqueeze(1).expand(batch_size, n_agents, -1).reshape(batch_size * n_agents, -1)
        
        # Encode all states at once
        with torch.no_grad():
            batch_agent_embeddings = self.encoder(batch['individual_states'].view(-1, self.individual_state_dim)).view(batch_size, n_agents, -1)
        
        # Compute role aggregations
        batch_role_agg = torch.stack([
            self.role_manager.compute_role_aggregations(batch_agent_embeddings[b], batch['role_assignments'][b], method='mean')
            for b in range(batch_size)
        ])
        flat_role_agg = batch_role_agg.unsqueeze(1).expand(batch_size, n_agents, self.config.n_roles, -1).reshape(batch_size * n_agents, self.config.n_roles, -1)
        
        # Process by role
        total_actor_loss = 0.0
        for role_id in range(self.config.n_roles):
            role_mask = (flat_role_assignments == role_id)
            if not role_mask.any():
                continue
            
            role_states = flat_states[role_mask]
            role_env_states = flat_env_states[role_mask]
            role_agg_data = flat_role_agg[role_mask]
            
            actor = self.role_manager.get_actor(role_id)
            critic = self.role_manager.get_critic(role_id)
            alpha = self.temp_manager.get_alpha(role_id)
            
            # Sample actions from policy
            actions, log_probs = actor.sample(role_states, deterministic=False)
            
            # Compute Q values
            with torch.no_grad():
                q1, q2 = critic(role_states, actions, role_env_states, role_agg_data)
                q_values = torch.min(q1, q2).squeeze(-1)
            
            # Actor loss
            role_actor_loss = (alpha.detach() * log_probs - q_values).mean()
            total_actor_loss += role_actor_loss
            
            losses[f'actor_role_{role_id}'] = role_actor_loss.item()
            losses[f'actor_role_{role_id}_entropy'] = -log_probs.mean().item()
        
        # Backpropagation
        if total_actor_loss != 0:
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            
            if self.config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.role_manager.get_actor_parameters(), self.config.clip_grad_norm)
            
            self.actor_optimizer.step()
            losses['actor_total'] = total_actor_loss.item()
        
        return losses

    def _update_alphas(self, batch: Dict) -> Dict[str, float]:
        """
        Update temperature parameters for all roles. Vectorized version.
        """
        losses = {}
        
        batch_size = batch['individual_states'].shape[0]
        n_agents = batch['individual_states'].shape[1]
        
        # Reshape
        flat_states = batch['individual_states'].view(batch_size * n_agents, -1)
        flat_role_assignments = batch['role_assignments'].view(batch_size * n_agents)
        
        # Process by role
        role_alpha_losses = {}
        for role_id in range(self.config.n_roles):
            role_mask = (flat_role_assignments == role_id)
            if not role_mask.any():
                continue
            
            role_states = flat_states[role_mask]
            actor = self.role_manager.get_actor(role_id)
            
            # Sample to get log probs
            with torch.no_grad():
                _, log_probs = actor.sample(role_states, deterministic=False)
            
            # Compute alpha loss
            alpha_loss = self.temp_manager.compute_alpha_loss(role_id, log_probs)
            role_alpha_losses[role_id] = alpha_loss
            
            losses[f'alpha_role_{role_id}'] = alpha_loss.item()
            losses[f'alpha_role_{role_id}_value'] = self.temp_manager.get_alpha(role_id).item()
        
        # Update all alphas
        if role_alpha_losses:
            alpha_loss_values = self.temp_manager.update(role_alpha_losses)
            losses['alpha_total'] = sum(alpha_loss_values.values())
        
        return losses

    def _update_vq_module(self, batch: Dict) -> Dict[str, float]:
        """
        Update VQ module (encoder and codebook). Vectorized version.
        """
        losses = {}
        
        batch_size = batch['individual_states'].shape[0]
        n_agents = batch['individual_states'].shape[1]
        
        # Process all states at once
        flat_states = batch['individual_states'].view(batch_size * n_agents, -1)
        
        # VQ forward pass for all states (batched version)
        # Process all states in one batch instead of loop
        # The VQ module returns 4 values when compute_loss=True
        role_ids, embeddings, quantized_embeddings, vq_losses = self.vq_module(flat_states, compute_loss=True)
        
        # Compute VQ losses
        total_vq_loss, vq_loss_dict = self.vq_loss.total_vq_loss(embeddings, quantized_embeddings)
        
        # Perplexity
        perplexity = self.vq_loss.compute_perplexity(role_ids, self.config.n_roles)
        
        # Backpropagation
        self.vq_optimizer.zero_grad()
        total_vq_loss.backward()
        
        if self.config.clip_grad_norm is not None:
            vq_params = list(self.encoder.parameters()) + list(self.vq_module.codebook.parameters())
            torch.nn.utils.clip_grad_norm_(vq_params, self.config.clip_grad_norm)
        
        self.vq_optimizer.step()
        
        # Record
        losses['vq_codebook'] = vq_loss_dict['codebook_loss'].item()
        losses['vq_commitment'] = vq_loss_dict['commitment_loss'].item()
        losses['vq_total'] = total_vq_loss.item()
        losses['vq_perplexity'] = perplexity.item()
        
        # Codebook usage
        usage = self.vq_module.get_codebook_usage()
        for i, u in enumerate(usage):
            losses[f'codebook_usage_role_{i}'] = u
        
        return losses

    def _process_observations(self, obs: Any) -> torch.Tensor:
        """Convert observations to tensor."""
        if isinstance(obs, tuple):
            # Multi-agent observations - convert to numpy array first
            obs_array = np.array([np.array(o, dtype=np.float32) for o in obs], dtype=np.float32)
            obs_tensor = torch.from_numpy(obs_array).to(self.device)
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        return obs_tensor
    
    def _compute_env_state(self, individual_states: torch.Tensor) -> torch.Tensor:
        """Compute global environment state from individual states."""
        # Simple concatenation
        env_state = individual_states.flatten()
        return env_state
    
    def _log_training_progress(self, start_time: float):
        """Log training progress."""
        elapsed_time = time.time() - start_time
        steps_per_sec = self.total_steps / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n[Step {self.total_steps}]")
        print(f"  Episodes: {self.episode_count}")
        print(f"  Steps/sec: {steps_per_sec:.1f}")
        
        if self.episode_returns:
            print(f"  Mean return (100ep): {np.mean(self.episode_returns):.2f}")
            print(f"  Mean length (100ep): {np.mean(self.episode_lengths):.1f}")
        
        # Buffer stats
        buffer_stats = self.replay_buffer.get_statistics()
        print(f"  Buffer size: {buffer_stats['size']}")
    
    def evaluate(self, n_episodes: int = 5) -> float:
        """
        Evaluate current policy.
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            Mean episode return
        """
        print(f"\nEvaluating for {n_episodes} episodes...")
        
        self.role_manager.eval_mode()
        
        episode_returns = []
        
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            episode_return = 0.0
            done = False
            
            while not done:
                # Select actions deterministically
                individual_states = self._process_observations(obs)
                actions, _ = self._select_actions(individual_states)
                
                obs, rewards, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                
                # Convert rewards from dict to array if needed
                if isinstance(rewards, dict):
                    rewards_sum = sum(rewards.values())
                else:
                    rewards_sum = np.sum(rewards)
                
                episode_return += rewards_sum
            
            episode_returns.append(episode_return)
        
        self.role_manager.train_mode()
        
        mean_return = np.mean(episode_returns)
        print(f"  Eval return: {mean_return:.2f} (+/- {np.std(episode_returns):.2f})")
        
        return mean_return
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.save_dir / filename
        
        checkpoint = {
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'best_eval_return': self.best_eval_return,
            'config': self.config.to_dict(),
            'vq_module': self.vq_module.state_dict(),
            'encoder': self.encoder.state_dict(),
            'role_manager': {
                'actors': {i: self.role_manager.get_actor(i).state_dict() 
                          for i in range(self.config.n_roles)},
                'critics': {i: self.role_manager.get_critic(i).state_dict()
                           for i in range(self.config.n_roles)},
                'target_critics': {i: self.role_manager.get_target_critic(i).state_dict()
                                  for i in range(self.config.n_roles)}
            },
            'temp_manager': {
                'log_alphas': [self.temp_manager.log_alphas[i].data 
                              for i in range(self.config.n_roles)],
                'target_entropies': self.temp_manager.target_entropies
            },
            'optimizers': {
                'actor': self.actor_optimizer.state_dict(),
                'critic': self.critic_optimizer.state_dict(),
                'vq': self.vq_optimizer.state_dict()
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint_path = self.save_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        self.best_eval_return = checkpoint['best_eval_return']
        
        self.vq_module.load_state_dict(checkpoint['vq_module'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        
        # Load role manager networks
        for i in range(self.config.n_roles):
            self.role_manager.get_actor(i).load_state_dict(
                checkpoint['role_manager']['actors'][i]
            )
            self.role_manager.get_critic(i).load_state_dict(
                checkpoint['role_manager']['critics'][i]
            )
            self.role_manager.get_target_critic(i).load_state_dict(
                checkpoint['role_manager']['target_critics'][i]
            )
        
        # Load temp manager
        for i in range(self.config.n_roles):
            self.temp_manager.log_alphas[i].data = checkpoint['temp_manager']['log_alphas'][i]
        self.temp_manager.target_entropies = checkpoint['temp_manager']['target_entropies']
        
        # Load optimizers
        self.actor_optimizer.load_state_dict(checkpoint['optimizers']['actor'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizers']['critic'])
        self.vq_optimizer.load_state_dict(checkpoint['optimizers']['vq'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from step {self.total_steps}")