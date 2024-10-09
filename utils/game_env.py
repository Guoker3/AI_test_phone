import time

from utils.MT_device import *


class GameEnv():
    def __init__(self, device):
        self.UI_device = device.UI_device

    def into_1v1_from_mainpage(self, restart_app=True):
        if restart_app:
            self.UI_device.app_stop('com.tencent.tmgp.sgame')
            time.sleep(10)
            self.UI_device.app_start('com.tencent.tmgp.sgame')
            self.UI_device.paddle_ocr_wait_appear(text_wait_for='开始游戏', time_out=180, click=True)
        self.UI_device.paddle_ocr_click('对战')
        self.UI_device.paddle_ocr_click('单人训练')
        self.UI_device.paddle_ocr_click('单人训练')
        self.UI_device.opencv_find_click('expand_hero_pool.png')
        self.UI_device.paddle_ocr_click('对抗路')
        self.UI_device.opencv_find_click('yase_icon.png')
        self.UI_device.paddle_ocr_click('挑选对手')
        self.UI_device.opencv_find_click('expand_hero_pool.png')
        self.UI_device.opencv_find_click('yase_icon.png')
        self.UI_device.paddle_ocr_click('开始对战')
        self.UI_device.paddle_ocr_wait_appear(text_wait_for='敌方英雄', time_out=180)
        self.UI_device.paddle_ocr_click('无敌', index=0)
        self.UI_device.opencv_find_click('close_1v1_control_panal.png')


if __name__ == '__main__':
    serials = '770b1af'
    device = MT_Device(serials)
    game = GameEnv(device)
    game.into_1v1_from_mainpage()
    device.stop()
