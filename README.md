# off_policy_ppo

在PPO算法中使用DQN中的replay buffer（使用(state, action, reward, next_state)训练，而不必等到回合结束）
advantage_t = q(s_t,a_t) - mean(q(s_t,a_t)) t=1,...,batch_size
