"""CTS verifier device implementation."""
import torch
# -*- coding: utf-8 -*-
from loguru import logger
import logging
import re
import time
import uiautomator2
import os, sys
import tempfile
from subprocess import Popen, PIPE

import aircv as ac
from paddleocr import PaddleOCR, draw_ocr  # pip install --no-cache-dir paddlepaddle==2.5.1 paddleocr==2.7.0.3

import cv2
import matplotlib.pyplot as plt
import numpy as np

_TIMEOUT = 10
_STEP = 0.5


class UI_Device(uiautomator2.Device):
    def __init__(self, static_path, serials, gpu):
        super(UI_Device, self).__init__(serials)
        self.gpu = gpu
        self.screenshot_pid = 0

        self.air_folder = None
        self.ac = ac
        # OpenMP冲突，删除anaconda的libiomp5md.dll:https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/123998470
        self.ocr_ch = PaddleOCR(use_angle_cls=True,
                                lang="ch")  # need to run only once to download and load model into memory
        self.ocr_en = PaddleOCR(use_angle_cls=True, lang="en")
        self.static = static_path

    def opencv_find_click(self, src, delay=5):  # only for icon, useless for text images
        time.sleep(delay)
        screen_pict = cv2.imread(self.screen_shot())
        file_path = os.path.join(self.static, src)
        src_pict = cv2.imread(file_path)
        res = cv2.matchTemplate(screen_pict, src_pict, method=cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        self.click(max_loc[0], max_loc[1])

    def opencv_find_try_click(self, src, delay=10):  # only for icon, useless for text images
        time.sleep(delay)
        screen_pict = cv2.imread(self.screen_shot())
        src_pict = cv2.imread(src)
        try:
            res = cv2.matchTemplate(screen_pict, src_pict, method=cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            self.click(max_loc[0], max_loc[1])
        except Exception as e:
            pass

    def paddle_ocr_click(self, text: str, index=0, delay=5):
        time.sleep(delay)
        if '\u4e00' <= text[0] <= '\u9fff':
            ocr = self.ocr_ch
        elif ('a' <= text[0] <= 'z') or ('A' <= text[0] <= 'Z'):
            ocr = self.ocr_en
        else:
            raise Exception('paddle ocr input type error')
        time.sleep(delay)
        screen_pict = ac.imread(self.screen_shot())
        result = ocr.ocr(screen_pict, cls=True)[0]
        aim = [x for x in result if text in x[1][0]]
        aim = aim[index]
        pos = aim[0]
        x = (pos[0][0] + pos[1][0]) / 2
        y = (pos[0][1] + pos[2][1]) / 2
        self.click(x, y)
        return [x, y]

    def paddle_ocr_try_click(self, text: str, index=0):
        if '\u4e00' <= text[0] <= '\u9fff':
            ocr = self.ocr_ch
        elif ('a' <= text[0] <= 'z') or ('A' <= text[0] <= 'Z'):
            ocr = self.ocr_en
        else:
            raise Exception('paddle ocr input type error')
        time.sleep(10)
        screen_pict = ac.imread(self.screen_shot())
        try:
            result = ocr.ocr(screen_pict, cls=True)[0]
            aim = [x for x in result if text in x[1][0]]
            aim = aim[index]
            pos = aim[0]
            x = (pos[0][0] + pos[1][0]) / 2
            y = (pos[0][1] + pos[2][1]) / 2
            self.click(x, y)
        except Exception as e:
            pass

    def screen_shot(self):
        """Take a screenshot."""
        path = self._get_output_path('png')
        self.screenshot(path)
        # logging.info('Stored screenshot %s', path)
        return path

    def start_record(self, file, serials):
        try:
            cmd = 'scrcpy -s ' + serials + ' -Nr ' + file
            os.popen(cmd)
            time.sleep(5)
        except  Exception as e:
            print(e)

    def _get_output_path(self, extension):
        """Generate a path for a screenshot and XML dump."""
        filename = '%s_%d.%s' % (self.serial, self._now(), extension)
        return os.path.join(tempfile.mkdtemp(), filename)

    def _now(self):
        """Get the current time in epoch milliseconds. Visible for testing."""
        return int(time.time() * 1000)

    def exists_click(self, **kwargs):
        if self.exists(**kwargs):
            self(**kwargs).click()
            time.sleep(5)
            return True
        return False

    def skip_app_wizard(self, pkg_name):
        time.sleep(5)
        for i in range(15):
            if self.exists_click(resourceId='%s:id/permission_agree_btn' % pkg_name):
                continue
            else:
                break

    def skip_app_policy(self, pkg_name):
        time.sleep(5)
        self.exists_click(resourceId='%s:id/tv_agree' % pkg_name)

    def start_apk(self, pkg_name):
        self.app_start(pkg_name, stop=True)
        time.sleep(8)  # wait for app load
        self.air_folder = './static/%s/' % pkg_name.replace('.', '_')
        if pkg_name == 'com.sankuai.meituan':
            time.sleep(4)  # wait for advertise load
            self.back_to(step_number=5, description='我的')

    def grant_app_all_permission(self, pkg_name):
        logger.info('granting permissions for %s' % pkg_name)
        if self.package_exist(pkg_name):
            self.run_cmd(
                'adb -s %s shell pm grant %s android.permission.WRITE_EXTERNAL_STORAGE' % (self.serial, pkg_name))
            self.run_cmd(
                'adb -s %s shell pm grant %s android.permission.READ_EXTERNAL_STORAGE' % (self.serial, pkg_name))
            self.run_cmd('adb -s %s shell pm grant %s android.permission.CAMERA' % (self.serial, pkg_name))
            self.run_cmd('adb -s %s shell pm grant %s android.permission.SYSTEM_ALERT_WINDOW' % (self.serial, pkg_name))
            self.run_cmd('adb -s %s shell pm grant %s android.permission.WRITE_SYNC_SETTINGS' % (self.serial, pkg_name))
            self.run_cmd('adb -s %s shell pm grant %s android.permission.VASSIST_DESKTOP' % (self.serial, pkg_name))
            self.run_cmd('adb -s %s shell pm grant %s android.permission.GET_ACCOUNTS' % (self.serial, pkg_name))
            self.run_cmd('adb -s %s shell pm grant %s android.permission.POST_NOTIFICATIONS' % (self.serial, pkg_name))
            self.run_cmd(
                'adb -s %s shell pm grant %s android.permission.ACCESS_FINE_LOCATION' % (self.serial, pkg_name))
            self.run_cmd('adb -s %s shell pm grant %s android.permission.RECORD_AUDIO' % (self.serial, pkg_name))
            logger.info('granted')
        else:
            logger.info('%s not exist, end the granting, please check the app')

    def install_or_update_tp_app(self, pkg_name):  ##TODO 因服务升级，旧版应用详情页已停止使用
        self.run_cmd('adb shell am start -a android.intent.action.VIEW -d market://details?id=%s' % pkg_name)
        time.sleep(5)
        if self.try_click_find(resourceId='com.xiaomi.market:id/tv_positive') is not None:
            time.sleep(5)
            if self.exists(resourceId='com.xiaomi.market:id/empty_detail_view_tv'):
                self.run_cmd('adb shell am start -a android.intent.action.VIEW -d market://details?id=%s' % pkg_name)
        if self.try_click_find(description='安装') is not None:
            self.wait_appear(time_out=30, description='打开')
            self.write_log.info('%s install success' % pkg_name)

    def find(self, xpath, timeout_secs=_TIMEOUT, step_secs=_STEP, **kwargs):
        """Looks for an object that matches the selectors, raising if not found."""
        obj = self._find(xpath, timeout_secs, step_secs, **kwargs)
        if obj is None:
            raise RuntimeError('No object matching %s %s' % (xpath, kwargs))
        return obj

    def try_find(self, xpath='', timeout_secs=_TIMEOUT, step_secs=_STEP, **kwargs):
        """Looks for an object that matches the selectors, raising if not found."""
        obj = None
        time.sleep(2)
        if self.exists(**kwargs):
            obj = self._find(xpath, timeout_secs, step_secs, **kwargs)
        return obj

    def try_click_find(self, xpath='', timeout_secs=_TIMEOUT, step_secs=_STEP, **kwargs):
        """Looks for an object that matches the selectors,if exists then click, raising if not found."""
        obj = None
        time.sleep(1)
        if self.exists(**kwargs):
            obj = self._find(xpath, timeout_secs, step_secs, **kwargs)
            obj.click()
            time.sleep(1.5)
        else:
            time.sleep(1)
            if self.exists(**kwargs):
                obj = self._find(xpath, timeout_secs, step_secs, **kwargs)
                obj.click()
                time.sleep(1.5)

        return obj

    def skip_danger(self):
        time.sleep(1.5)
        if self.exists(text='Danger') or self.exists(text='Attention') or self.exists(text='Important warning'):
            time.sleep(2)
            count = 36
            flag = True
            while count > 0:
                count -= 1
                time.sleep(5)
                if self.exists(text='Danger') or self.exists(text='Attention') or self.exists(text='Important warning'):
                    if flag:
                        self.try_click_snooze_find('', resourceId="com.miui.securitycenter:id/check_box")
                        flag = False
                        time.sleep(2)
                    if self.exists(text="OK"):
                        self.wait_appear(time_out=30, text="OK", enabled='true')
                        self.try_click_find('', text="OK", enabled='true')
                        flag = True
                    elif self.exists(text="Next"):
                        self.wait_appear(time_out=30, text="Next", enabled='true')
                        self.try_click_find('', text="Next", enabled='true')
                        flag = True
                    elif self.exists(text="Accept"):
                        self.wait_appear(time_out=30, text="Accept", enabled='true')
                        self.try_click_find('', text="Accept", enabled='true')
                        flag = True
                else:
                    break

            time.sleep(2)

    def _ret_keywords(self, flag):
        if flag == 'confirm':
            key_words = ["同意", "确定", "同意并继续", "仅在使用中允许", "com.android.camera:id/cvtype_btn_select_cv"]
            # key_words = ["Wait", "Continue", "Connect", "Confirm", "CONFIRM", "Next", "OK", "Ok", "GOT IT", "Got it",
            #              "Got It", "Agree", "ALLOW",
            #              "Allow", "同意", "确定", "Accept", "ACCEPT", 'Accept & continue', "TURN ON", "Turn on", "INSTALL",
            #              "Install", "UPDATE"]
            # key_words.extend(
            #     ['Always allow', 'WHILE USING THE APP', 'While using the app', 'ONLY THIS TIME', 'Only this time',
            #      "Allow all the time",
            #      "Allow only while using the app"])
            # key_words.extend(['DONE', ])
            # key_words.append('com.lbe.security.miui:id/permission_allow_button_1')
        elif flag == 'reject':
            key_words = ["Deny", "CLOSE", "Close", "CANCEL", "Cancel", "No thanks"]
            key_words.extend(["DON'T ALLOW", "Don’t allow"])
            key_words.append('com.lbe.security.miui:id/permission_deny_button_1')
            key_words.append('com.android.camera:id/btn_cancel')
            key_words.extend(['Dismiss', ])
            key_words.extend(['Decline', 'DECLINE'])
        else:
            raise Exception('flag of keywords for skip not set')
        return key_words

    def skip_confirm(self, loop_limit=1):
        time.sleep(8)
        self.skip_page('', self._ret_keywords('confirm'), loop_limit)
        time.sleep(8)

    def skip_reject(self, loop_limit=1):
        time.sleep(2)
        self.skip_page('', self._ret_keywords('reject'), loop_limit)
        time.sleep(2)

    def skip_page(self, xpath='', key_words=None, loop_limit=1, timeout_secs=_TIMEOUT, step_secs=_STEP):
        """detect page pop up accidently and try to skip it"""
        time.sleep(8)
        if self.exists(resourceId='com.miui.securityinputmethod:id/dropdown'):
            self.snooze_find('', resourceId='com.miui.securityinputmethod:id/dropdown').click()
        if key_words is None or key_words == list():
            key_words = list()
            key_words.extend(self._ret_keywords('confirm'))
            key_words.extend(self._ret_keywords('reject'))

        obj = None
        for button in key_words:
            if key_words.count('.') < 2:
                if self.exists(text=button):
                    obj = self._find(xpath, timeout_secs, step_secs, text=button, clickable=True)
            else:
                if self.exists(resourceId=button):
                    obj = self._find(xpath, timeout_secs, step_secs, resourceId=button, clickable=True)
            if obj is not None:
                # self.screen_shot()
                obj.click()
                if loop_limit > 0:
                    time.sleep(2)
                    self.skip_page(xpath, key_words, loop_limit - 1, timeout_secs, step_secs)
                return True

        # self.click(0.048, 0.021)
        return False

    def while_click(self, words_list, device_list, word_type='text', time_out=15, max_time=300):
        if type(device_list) != type(list()):
            device_list = [device_list, ]
        t = time.time()
        t_start = time.time()
        loop = 0
        flag = False
        while (time.time() - t < time_out and time.time() - t_start < max_time) or loop <= 1:
            loop += 1
            for w in words_list:
                for d in device_list:
                    if word_type == 'text':
                        try:
                            if d.exists(text=w):
                                d.find('', text=w).click()
                                t = time.time()
                                flag = True
                        except Exception:
                            pass
                    elif word_type == 'resourceId':
                        try:
                            if d.exists(resourceId=w):
                                d.find('', resourceId=w).click()
                                t = time.time()
                                flag = True
                        except Exception:
                            pass
                    elif word_type == 'all':
                        if self.skip_page(key_words=words_list):
                            t = time.time()
                            flag = True
        return flag

    def snooze_find(self, xpath='', timeout_secs=_TIMEOUT, step_secs=_STEP, **kwargs):
        """loop and ooks for an object that matches the selectors."""
        count_flag = 8
        skip_limit = 5
        obj = None
        time.sleep(5)
        while count_flag >= 0 and obj is None:
            time.sleep(5)
            if kwargs is not None:
                if self.exists(**kwargs):
                    obj = self.find(xpath, timeout_secs, step_secs, **kwargs)
            else:
                obj = self.find(xpath, timeout_secs, step_secs)
            count_flag -= 1
            if count_flag == 4 and self.exists(scrollable=True):
                if self.exists(scrollable=True, resourceId='android:id/list'):
                    self(scrollable=True, resourceId='android:id/list').scroll.vert.forward(max_swipes=4, **kwargs)
                elif self.exists(scrollable=True, resourceId='com.miui.securitycenter:id/view_pager'):
                    self(scrollable=True, resourceId='com.miui.securitycenter:id/view_pager').scroll.vert.forward(
                        max_swipes=4,
                        **kwargs)
                elif self.exists(scrollable=True, resourceId='com.android.settings:id/recycler_view'):
                    self(scrollable=True, resourceId='com.android.settings:id/recycler_view').scroll.vert.forward(
                        max_swipes=4,
                        **kwargs)
                else:
                    self(scrollable=True).scroll.vert.forward(max_swipes=1, **kwargs)
                obj = self.try_find('', **kwargs)
            ##TODO  very important ,skip page with AI
            # if count_flag == 0:
            #     if (self.skip_page(xpath='', key_words=None, loop_limit=1)) and skip_limit > 0:
            #         skip_limit -= 1
            #         count_flag = 8
        if obj is None:
            #  if 'text' in kwargs:
            #       if kwargs['text']=="PASS":
            #            self.press('back')
            #    if self.exists(text="PASS"):
            #        return self.find('',text='PASS')
            #    else:
            raise Exception('No object matching %s %s' % (xpath, kwargs))
        # obj=False
        # time.sleep(0.75)
        return obj

    def try_click_snooze_find(self, xpath='', timeout_secs=_TIMEOUT, step_secs=_STEP, **kwargs):
        try:
            obj = self.snooze_find(xpath, timeout_secs, step_secs, **kwargs)
        except Exception:
            # obj=None
            return False
        if obj is not None:
            obj.click()
            return True

    def find_button_or(self, text_list):
        time.sleep(2)
        for i in text_list:
            if self.exists(text=i):
                return self(text=i)
        return False

    def catch_find(self, wait_time=5, click=False, **kwargs):
        t = time.time()
        while time.time() - t < wait_time:
            if self.exists(**kwargs):
                if click:
                    self.try_click_find(**kwargs)
                return True
            else:
                time.sleep(0.1)
                self.screen_on()
        return False

    def is_present(self, xpath, timeout_secs=_TIMEOUT, step_secs=_STEP, **kwargs):
        """Checks if an object matching the selectors exists."""
        return self._find(xpath, timeout_secs, step_secs, **kwargs) is not None

    def light_up(self):
        for i in range(2):
            time.sleep(0.5)
            if not self.info.get('screenOn'):
                self.press("power")
                break
            else:
                break

    def check_exists(self, time_step=1, **kwargs):
        time.sleep(time_step)
        if self.exists(**kwargs):
            return True
        time.sleep(time_step)
        if self.exists(**kwargs):
            return True
        return False

    def wait_starting_up(self):
        time.sleep(30)
        t = time.time()
        while True and time.time() - t < 600:
            res = os.popen('adb -s {} shell getprop init.svc.bootanim'.format(self.serial)).read()
            if 'stop' in res:
                break
            else:
                time.sleep(5)

    def paddle_ocr_wait_appear(self, text_wait_for, time_out=60, click=False):
        t = time.time()
        while time.time() - t < time_out:
            time.sleep(5)
            if '\u4e00' <= text_wait_for[0] <= '\u9fff':
                ocr = self.ocr_ch
            elif ('a' <= text_wait_for[0] <= 'z') or ('A' <= text_wait_for[0] <= 'Z'):
                ocr = self.ocr_en
            else:
                raise Exception('paddle ocr input type error')
            screen_pict = ac.imread(self.screen_shot())
            try:
                result = ocr.ocr(screen_pict, cls=True)[0]
            except Exception as e:
                continue
            aim = [x for x in result if text_wait_for in x[1][0]]
            aim = aim[0]
            pos = aim[0]
            x = (pos[0][0] + pos[1][0]) / 2
            y = (pos[0][1] + pos[2][1]) / 2
            if click:
                self.click(x, y)
            return True
        return False

    def wait_appear(self, time_out=30, click=False, skip=True, **kwargs):
        t = time.time()
        count = 0
        while time.time() - t < time_out:
            count += 1
            if skip and count % 7 == 0:
                self.screen_on()
                self.skip_page()
            if count % 11 == 0:
                time.sleep(3)
            time.sleep(0.75)
            if self.exists(**kwargs):
                if click:
                    self.find('', **kwargs).click()
                return True
        return False

    def wait_disappear(self, time_out=120, check_time=2, skip=True, **kwargs):
        t = time.time()
        time.sleep(2)
        if check_time > 2:
            check_time_2 = 2
        else:
            check_time_2 = check_time
        while time.time() - t < time_out:
            self.screen_on()
            if not self.exists(**kwargs):
                time.sleep(check_time_2)
                if skip:
                    self.skip_page()
                if not self.exists(**kwargs):
                    return True
            time.sleep(check_time)
        return False

    def wait_for_absent(self, timeout_secs=_TIMEOUT, step_secs=_STEP, **kwargs):
        time_start = time.time()
        while self.is_present('', 1, step_secs, **kwargs):
            time.sleep(1)
            time_stop = time.time()
            if int(time_stop - time_start) > timeout_secs:
                raise RuntimeError('TimeOut')

    def _find(self, xpath, timeout_secs, step_secs, **kwargs):
        """Looks for an object that matches the selectors.

        Args:
          selectors: selector keys or values to look for
          timeout_secs: maximum time to look for
          step_secs: seconds to wait between checks
          **kwargs: additional selectors to use
        Returns:
          UI object that matches the selector or None.
        """
        logging.debug('Looking for object matching %s and %s', xpath, kwargs)
        # Periodically check existence until timeout exceeded.
        end_time = time.time() + timeout_secs
        while time.time() < end_time:
            if xpath:
                # def __call__(self, xpath: str, source=None):
                #   # print("XPATH:", xpath)
                #   return XPathSelector(self, xpath, source)
                obj = self.xpath(xpath)
            else:
                # def __call__(self, **kwargs):
                #   return UiObject(self, Selector(**kwargs))
                obj = self(**kwargs)
            if obj.exists:
                return obj
            time.sleep(step_secs)

    def run_cmd(self, cmd_string, timeout=20):
        logging.info('Running ADB command \'%s\' ' % (cmd_string))
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

    def app_install(self, args):
        logging.info(self.run_cmd('adb -s %s install %s' % (self.serial, args)))
        time.sleep(2)
        if self.exists(text='Google Play Protect'):
            self.try_click_find(text='More details')
            time.sleep(2)
            a = self.find('', text='Got it')
            self.find('', text='Got it').click(offset=(-2, -15.5 / 6))
        time.sleep(14)

    def app_uninstall(self, args):
        logging.info(self.run_cmd('adb -s %s uninstall %s' % (self.serial, args)))
        time.sleep(2)

    def adb_reboot(self):
        logging.info(self.run_cmd('adb -s %s reboot' % self.serial))
        self.wait_starting_up()

    def adb_push(self, args):
        logging.info(self.run_cmd('adb -s %s push %s' % (self.serial, args)))
        time.sleep(4)

    def adb_send_keys(self, args):
        time.sleep(2)
        logging.info(self.run_cmd('adb -s %s shell input text %s' % (self.serial, args)))
        time.sleep(5)

    def bt_name(self):
        serial = self.serial
        name = 'bt' + serial
        return name

    def enable_bm(self, text):
        self.swipe(0.9, 0.01, 0.9, 0.5)
        if self.is_present('', timeout_secs=1, resourceId="android:id/button1", text='Got it'):
            self(resourceId="android:id/button1", text='Got it').click()
        time.sleep(2)
        bt_status = self(resourceId="com.android.systemui:id/title", text=text).sibling(
            resourceId="com.android.systemui:id/status").get_text()
        if bt_status == 'Off':
            self.find('', resourceId="com.android.systemui:id/title", text=text).click()
        elif bt_status == 'Not available':
            raise EnvironmentError("Please insert sim card!")
        self.press('back')

    def disable_bm(self, text):
        self.swipe(0.9, 0.01, 0.9, 0.5)
        if self.is_present('', timeout_secs=1, resourceId="android:id/button1", text='Got it'):
            self(resourceId="android:id/button1", text='Got it').click()
        time.sleep(2)
        bt_status = self(resourceId="com.android.systemui:id/title", text=text).sibling(
            resourceId="com.android.systemui:id/status").get_text()
        if bt_status == 'On':
            self.find('', resourceId="com.android.systemui:id/title", text=text).click()
        self.press('back')

    def enable_location(self):
        self.run_cmd('adb -s %s shell cmd location set-location-enabled true' % self.serial)
        # self.logic_UI_open('location')
        # time.sleep(2)
        # self.enable_all_settings_button_exist()
        # self.open_cts_verifier()

    def disable_location(self):
        self.run_cmd('adb -s %s shell cmd location set-location-enabled false' % self.serial)
        # self.logic_UI_open('location')
        # time.sleep(2)
        # if self.exists(text='No apps have requested location recently', enabled='true'):
        #     self.snooze_find(resourceId='android:id/widget_frame').click()
        # self.open_cts_verifier()

    def enable_wifi(self, continue_connect=None):

        self.run_cmd('adb -s %s shell svc wifi enable' % self.serial)
        time.sleep(2)

        # self.logic_UI_open('wifi')
        # time.sleep(2)
        # if not self.exists(text='Available networks'):
        #     self.snooze_find(resourceId='android:id/widget_frame').click()
        # time.sleep(2)
        # if continue_connect is not None:
        #     self.snooze_find(text=continue_connect).click()
        #     time.sleep(3)
        # self.open_cts_verifier()

    def disable_wifi(self):
        self.run_cmd('adb -s %s shell svc wifi disable' % self.serial)
        time.sleep(2)
        # self.logic_UI_open('wifi')
        # time.sleep(2)
        # if self.exists(text='Available networks'):
        #     self.snooze_find(resourceId='android:id/widget_frame').click()
        # time.sleep(2)
        # self.open_cts_verifier()

    def click_continuous(self, number, **kwargs):
        obj = self.snooze_find('', **kwargs)
        bounds = obj.info['bounds']
        x = (bounds['left'] + bounds['right']) / 2
        y = (bounds['top'] + bounds['bottom']) / 2
        for i in range(number):
            self.touch.down(x, y).up(x, y)

    def current_info(self):
        '''
        adb shell dumpsys window | grep mFocusedWindow
        output:
         mFocusedWindow=Window{dedd5ca u0 com.miui.gallery/com.miui.gallery.activity.HomePageActivity}
        RE:
         r'mFocusedWindow=Window{.*\s+(?P<package>[^\s]+)/(?P<activity>[^\s]+)}'
        :return: str activity
        '''
        time.sleep(0.5)
        _RE1 = re.compile(r'mFocusedWindow=Window{.*\s+(?P<info>[^\s]+)}')
        # _RE2 = re.compile(r'mFocusedWindow=Window{.*\s+(?P<package>[^\s]+)/(?P<activity>[^\s]+)}')
        # mFocusedWindow=Window{ed5ed0c mode=0 rootTaskId=1 u0 NotificationShade}
        output = _RE1.search(self.shell('dumpsys window | grep mFocusedWindow')[0])
        logging.info(output)
        if output:
            info = output.group('info')
            if '/' in info:
                pac = info.split('/')[0]
                act = info.split('/')[1]
                return dict(package=pac, activity=act)
            else:
                return dict(package=info, activity=info)
        raise OSError("Couldn't get focused app")

    def current_activity(self):
        return self.current_info()['activity']

    def current_package(self):
        return self.current_info()['package']

    def getprop(self, prop):
        '''
        :param prop: adb shell getprop | grep prop
        :return: str prop
        '''
        output = self.shell('getprop | grep %s' % prop)[0]
        _RE = re.compile(r':\s*\[(\S+)\]')
        prop = _RE.search(output).group(1)
        return prop

    def build_region(self):
        return self.getprop('ro.miui.build.region')

    def device_name(self):
        try:
            name = self.getprop('ro.product.device')
        except AttributeError:
            # 'NoneType' object has no attribute 'group'
            name = ''
        return name

    def get_fingerprint(self):
        return self.getprop('ro.build.fingerprint')

    def get_api(self):
        return int(self.getprop('ro.build.version.sdk'))

    def get_version(self):
        output = self.get_fingerprint()
        version = output.split('/')[-2].split(':')[0]
        return version

    def package_exist(self, package_name):
        '''
        :param package_name: adb shell pm list packages | grep package_name
        :return: if return null return False, else return True
        '''
        output = self.shell('pm list packages | grep %s' % package_name)[0]
        if output == '':
            return False
        else:
            return True

    def is_poco(self):
        return self.package_exist('com.mi.android.globallauncher')

    def scrcpy_record(self, file):
        try:
            cmd = 'scrcpy -s ' + self.serial + ' --no-window -Nr ' + file
            os.popen(cmd)
            time.sleep(5)
        except  Exception as e:
            print(e)

    def _scrcpy_screenshot(self, file):
        try:
            # (1080,2400,3)
            cmd = 'scrcpy -s ' + self.serial + ' --no-window --record-format=mkv  --record ' + file
            os.popen(cmd)
        except  Exception as e:
            print(e)

    def scrcpy_record_stop(self):
        time.sleep(5)
        if sys.platform.startswith('win'):
            rerad_content = os.popen('wmic process get name,processid,commandline | findstr scrcpy').readlines()
            for item in rerad_content:
                if 'scrcpy-server.jar' in item and self.serial in item:
                    b = item.strip().split(' ')
                    self.serial = b[2]
                    pid = b[-1].strip()
                    cmd = 'taskkill -PID ' + pid + ' -F'
                    os.popen(cmd)
        else:
            read_content = os.popen('ps aux | grep scrcpy').readlines()
            for item in read_content:
                if 'scrcpy-server.jar' in item and self.serial in item:
                    b = item.strip().split()
                    pid = b[1].strip()
                    cmd = 'kill ' + pid
                    os.popen(cmd)

    def get_screen(self):
        temp_pict_name = '%s_%d_%d.%s' % (self.serial, self.screenshot_pid, int(time.time() * 1000), 'png')
        temp_pict = os.path.join(tempfile.mkdtemp(), temp_pict_name)
        self.screenshot_pid += 1
        self._scrcpy_screenshot(temp_pict)
        screen_pict_raw = cv2.imread(self.screen_shot())  # OpenCV的读取顺序为B，G，R
        # 画面尺寸： (1080,2400,3)
        screen_pict = cv2.resize(screen_pict_raw, (1080, 2400))
        # 人物位置:  (9.7/20*1080,18.5/37.9*2400)=(651,1171)
        # plt.imshow(screen_pict)
        # plt.show()
        return screen_pict

    def get_screen_torch(self):
        screen = self.get_screen()
        screen_np = np.asarray(screen)
        screen_torch = torch.from_numpy(screen_np).cuda(self.gpu).unsqueeze(0).permute(0, 3, 2, 1) / 255
        return screen_torch

    def cut_image(self, img, left, upper, right, lower):
        """
            原图与所截区域相比较
        :param path: 图片路径
        :param left: 区块左上角位置的像素点离图片左边界的距离
        :param upper：区块左上角位置的像素点离图片上边界的距离
        :param right：区块右下角位置的像素点离图片左边界的距离
        :param lower：区块右下角位置的像素点离图片上边界的距离
        """
        cropped = img[int(upper):int(lower), int(left):int(right)]
        return cropped
        # plt.imshow(cropped)
        # plt.show()

    def mask_red_lane(self, hero_position=[0.485, 0.488], canvas_range=[1080, 2400]):
        img = self.get_screen()
        # the red color range of lane(RGB #7D221E,#D22724)
        # lower_bound = np.array([30, 34, 125])  # BGR
        # upper_bound = np.array([36, 39, 210])
        lower_bound = np.array([0, 0, 110])  # BGR
        upper_bound = np.array([50, 50, 240])
        # this gives you the mask for those in the ranges you specified,
        # but you want the inverse, so we'll add bitwise_not...
        red_img = cv2.inRange(img, lower_bound, upper_bound)
        # R_img = img[:, :, 2]  ##TODO not useful by just split the R space， use more limit to recognize.
        # plt.imshow(red_img)
        # plt.show()
        ##TODO can use conv but it may hard to judge, so use preset critic first.
        step = 9
        stride = [int(x / (step + 1)) for x in canvas_range]
        hero_position = [int(hero_position[0] * canvas_range[0]), int(hero_position[1] * canvas_range[1])]
        red_map = np.zeros((step, step))
        red_value_max = stride[0] * stride[1] * 255
        for i in range(step):  # i: width
            for j in range(step):  # j: height,
                left = hero_position[1] - (step / 2 - i) * stride[1]
                right = hero_position[1] - (step / 2 - 1 - i) * stride[1]
                upper = hero_position[0] - (step / 2 - j) * stride[0]
                lower = hero_position[0] - (step / 2 - 1 - j) * stride[0]
                cropped = self.cut_image(red_img, left, upper, right, lower)
                red_value = np.sum(cropped) / red_value_max
                red_map[j][i] = np.float64(red_value)
                plt.subplot(step, step, i + step * j + 1)
                plt.imshow(cropped)
        # plt.show()
        return step, red_map


if __name__ == '__main__':
    serial = '770b1af'
    device = UI_Device(static_path='../static', serials=serial)
    device.judge_the_red()
    A = 1
