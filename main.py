import argparse
import trainer
import tester
import sys, os

if __name__ == '__main__':
    '''
    Arguments parsing
    '''
    parser = argparse.ArgumentParser(description='RobUSRL')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--seed', type=int, default=40, help='random seed')
    parser.add_argument('--replay_buffer_size', type=int, default=5e3, help='replay buffer size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_steps', type=int, default=5e6, help='number of steps')
    parser.add_argument('--exp_steps', type=int, default=3e6, help='number of steps')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor')
    parser.add_argument('--start_step', type=float, default=0, help='start step')
    parser.add_argument('--warming_up_steps', type=int, default=1e4, help='warming up steps')
    parser.add_argument('--train_freq', type=int, default=1, help='training frequency')
    parser.add_argument('--test_freq', type=int, default=8e3, help='testing frequency')
    parser.add_argument('--target_update_freq', type=int, default=5e3, help='target network update frequency')
    parser.add_argument('--checkpoint_freq', type=int, default=1e4, help='checkpoint frequency')
    parser.add_argument('--eps_train', type=float, default=0.99, help='initial exploration rate')
    parser.add_argument('--eps_train_final', type=float, default=0.05, help='final exploration rate')
    parser.add_argument('--lr_step_size', type=int, default=3e4, help='period of learning rate decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.85, help='decay rate of learning rate')
    parser.add_argument('--alpha', type=float, default=0.6, help='alpha for prioritized replay buffer')
    parser.add_argument('--beta_train', type=float, default=0.4, help='initial beta for prioritized replay buffer')
    parser.add_argument('--beta_final', type=float, default=1, help='final beta for prioritized replay buffer')
    parser.add_argument('--envs_num', type=int, default=16, help='number of environments')
    parser.add_argument('--log_path', type=str, default="experiment/icm_new_action_fix_2", help='log path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--hdf5_path', type=str, default=None, help='hdf5 path')
    parser.add_argument('--use_icm', type=bool, default=True, help='use icm or not')
    parser.add_argument('--render', type=bool, default=False, help='render or not')
    parser.add_argument('--video_id', type=int, default=0, help='folder id for saving videos, e.g. video_id=0 means saving videos in folder video_0')
    parser.add_argument('--th', type=int, default=3, help='threshold scanned bone number for a successful scan')
    parser.add_argument('--w_dis', type=float, default=0.8, help='weight for distance')
    parser.add_argument('--w_percentage', type=float, default=0.2, help='weight for percentage')
    parser.add_argument('--test_episodes', type=int, default=100, help='number of test episodes')
    parser.add_argument('--tumor_size', type=str, default='all', help='size of tumors, e.g. small, medium, large, all')

    args = parser.parse_args()

    mode = args.mode
    seed = args.seed
    replay_buffer_size = args.replay_buffer_size
    learning_rate = args.learning_rate
    num_steps = args.num_steps
    exp_steps = args.exp_steps
    batch_size = args.batch_size
    discount = args.discount
    warming_up_steps = args.warming_up_steps
    start_step = args.start_step
    train_freq = args.train_freq
    test_freq = args.test_freq
    target_update_freq = args.target_update_freq
    checkpoint_freq = args.checkpoint_freq
    eps_train = args.eps_train
    eps_train_final = args.eps_train_final
    alpha = args.alpha
    beta_train = args.beta_train
    beta_final = args.beta_final
    envs_num = args.envs_num
    log_path = args.log_path
    checkpoint_path = args.checkpoint_path
    hdf5_path = args.hdf5_path
    lr_step_size = args.lr_step_size
    lr_decay_rate = args.lr_decay_rate
    use_icm = args.use_icm
    render = args.render
    video_id = args.video_id    
    th = args.th
    w_dis = args.w_dis
    w_percentage = args.w_percentage
    test_episodes = args.test_episodes
    tumor_size = args.tumor_size

    # Create log folder
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    if mode == 'train':
        trainer.run(seed=seed,
                    replay_buffer_size=replay_buffer_size,
                    learning_rate=learning_rate,
                    num_steps=num_steps,
                    exp_steps=exp_steps,
                    batch_size=batch_size,
                    discount=discount,
                    warming_up_steps=warming_up_steps,
                    start_step=start_step,
                    train_freq=train_freq,
                    test_freq=test_freq,
                    target_update_freq=target_update_freq,
                    checkpoint_freq=checkpoint_freq,
                    eps_train=eps_train,
                    eps_train_final=eps_train_final,
                    alpha=alpha,
                    beta_train=beta_train,
                    beta_final=beta_final,
                    envs_num=envs_num,
                    log_path=log_path,
                    checkpoint_path=checkpoint_path,
                    hdf5_path=hdf5_path,
                    lr_step_size=lr_step_size,
                    lr_decay_rate=lr_decay_rate,
                    use_icm=use_icm,
                    th=th,
                    w_dis=w_dis,
                    w_percentage=w_percentage,
                    tumor_size=tumor_size)
    elif mode == 'test':
        tester.run(seed=seed,
                    learning_rate=learning_rate,
                    discount=discount,
                    target_update_freq=target_update_freq,
                    render=render,
                    video_id=video_id,
                    checkpoint_path=checkpoint_path,
                    use_icm=use_icm,
                    th=th,
                    w_dis=w_dis,
                    w_percentage=w_percentage,
                    test_episodes=test_episodes,
                    tumor_size=tumor_size)
    else:
        raise NotImplementedError
