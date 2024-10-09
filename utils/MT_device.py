from pyminitouch import MNTDevice
import random
from utils.UI_device import *


class MT_Device(MNTDevice, uiautomator2.Device):
    def __init__(self, device_serial, gpu, static='A:\workBench\WZAI\static'):
        self.serials = device_serial
        self.gpu = gpu
        self.start_MNT_server()
        super().__init__(self.serials)
        self.static = static
        self.max_x = int(self.connection.max_x)
        self.max_y = int(self.connection.max_y)
        self.ocr_ch = PaddleOCR(use_angle_cls=True,
                                lang="ch")  # need to run only once to download and load model into memory
        self.ocr_en = PaddleOCR(use_angle_cls=True, lang="en")

        self.UI_device = UI_Device(static, self.serials, gpu)

    def _point_position_weditor2minitouch(self, weditor_x_percent, weditor_y_percent, flutter=0):
        weditor_x_percent = self._flutter_value(weditor_x_percent, flutter)
        weditor_y_percent = self._flutter_value(weditor_y_percent, flutter)
        touch_x = int((1 - weditor_y_percent) * self.max_x)
        touch_y = int(weditor_x_percent * self.max_y)
        return touch_x, touch_y

    def _flutter_value(self, value, flutter):
        if flutter != 0:
            value = value * (1 + random.randint(-100 * flutter, 100 * flutter) / 100)
        return value

    def point(self, weditor_x_percent, weditor_y_percent):
        # L1-音量键向上
        touch_x, touch_y = self._point_position_weditor2minitouch(weditor_x_percent, weditor_y_percent)
        self.tap([(touch_x, touch_y)])

    def slide(self, weditor_start_percent: list, weditor_stop_percent: list, flutter=0.01, no_down=False, no_up=False):
        touch_start_x, touch_start_y = self._point_position_weditor2minitouch(weditor_start_percent[0],
                                                                              weditor_start_percent[1], flutter)

        touch_stop_x, touch_stop_y = self._point_position_weditor2minitouch(weditor_stop_percent[0],
                                                                            weditor_stop_percent[1], flutter)
        self.swipe(
            [(touch_start_x, touch_start_y), (touch_stop_x, touch_stop_y)],
            duration=100,
            pressure=500,
            no_down=no_down,
            no_up=no_up,
        )

    def start_MNT_server(self):
        self.run_cmd("adb kill-server")
        self.run_cmd("adb -s %s root" % self.serials)
        self.run_cmd(
            "adb -s %s shell am startservice --user 0  -a jp.co.cyberagent.stf.ACTION_START  -n jp.co.cyberagent.stf/.Service" % self.serials)

    def run_cmd(self, cmd_string, timeout=20):
        logger.info('Running ADB command \'%s\' ' % (cmd_string))
        try:
            p = Popen(cmd_string, stderr=PIPE, stdout=PIPE, shell=True)
            t_beginning = time.time()
            res_code = 0
            while True:
                if p.poll() is not None:
                    break
                seconds_passed = time.time() - t_beginning
                if timeout and seconds_passed > timeout:
                    p.terminate()  # 等同于p.kill()
                    msg = "Timeout ：Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
                    raise Exception(msg)
                time.sleep(0.1)
            msg = str(p.stdout.read().decode('utf-8'))
        except Exception as e:
            res_code = 200
            msg = "[ERROR]Unknown Error : " + str(e)
        return msg

    # def __del__(self):
    #     self.stop()


if __name__ == '__main__':
    serial = '356a7b7'
    device = MT_Device(serial)
    device.point(0.3, 0.755)
    device.stop()
