import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import sys
import os
sys.path.append(os.getcwd())

import utils
import TD3_wBC


envs2=(
	"halfcheetah-random-v2",
	"hopper-random-v2",
	"walker2d-random-v2",
	"halfcheetah-medium-v2",
	"hopper-medium-v2",
	"walker2d-medium-v2",
	"halfcheetah-medium-replay-v2",
	"hopper-medium-replay-v2",
	"walker2d-medium-replay-v2",
	"halfcheetah-medium-expert-v2",
	"hopper-medium-expert-v2",
	"walker2d-medium-expert-v2",
	)

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


def main(args):
	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")
	if not os.path.exists("./results"):
		os.makedirs("./results")
	if not os.path.exists("./infos"):
		os.makedirs("./infos")
	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")
	exist_files = os.listdir('./results/')
	if file_name + '.npy' in exist_files:
		print(f'{file_name} exit, skipping')
		return

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	# policy = TD3_BC.TD3_BC(**kwargs)
	policy = TD3_wBC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	if args.normalize:
		mean, std = replay_buffer.normalize_states()
	else:
		mean, std = 0, 1

	evaluations = []
	info_log = []
	for t in range(int(args.max_timesteps)):
		info = policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			info_log.append(info)
			print(f"Time steps: {t + 1}", info)
			evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
			np.save(f"./results/{file_name}", evaluations)
			np.save(f"./infos/{file_name}", info_log)
			if args.save_model: policy.save(f"./models/{file_name}")
	foo = 0


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_wBC")               # Policy name
	parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=20, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5, type=float)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	seeds = [0, 1, 10, 11, 12]
	for i in seeds:
		args.seed = i
		for j in range(len(envs2)):
			args.env = envs2[j]
			main(args)
