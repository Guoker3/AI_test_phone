from gym.envs.registration import register

register(
    id='WZ_1v1_env_v0',  # 环境名,版本号v0必须有
    entry_point='models.WZ_1v1_env:WZ1v1Env'  # 文件夹名.文件名:类名
)
