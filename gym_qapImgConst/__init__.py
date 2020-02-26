from gym.envs.registration import register

register(
    id='qapImgConst-v0',
    entry_point='gym_qapImgConst.envs:QapImgConstEnv',
)