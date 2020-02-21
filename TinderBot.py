import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from selenium import webdriver
import time
import requests
import json
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

class TinderBot:
    def __init__(self):

        # Paste your own chromedriver application path in the below excutable_path argument
        self.driver = webdriver.Chrome(executable_path='C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
        # This input is nessesary for Chrome to be used for the selenium library.
        
        self.i = int(len(open('ImageURL.txt', 'r').read().split('\n')))
        self.MLmodel = tf.keras.models.load_model('.\\MLmodelData\\')
        self.faceDetector = MTCNN(min_face_size=100)
        self.path = '.\\images\\'
        self.preferGirl = True

    def login(self):
        self.driver.get('https://tinder.com')
        time.sleep(7)
        login = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/div[2]/button')
        login.click()
        time.sleep(1)
        self.driver.switch_to.window(self.driver.window_handles[1])
        with open('Login.json') as f:
            loginData = json.load(f)
        email = self.driver.find_element_by_xpath('//*[@id="email"]')
        email.send_keys(loginData['email'])
        passwrd = self.driver.find_element_by_xpath('//*[@id="pass"]')
        passwrd.send_keys(loginData['password'])
        passwrd.submit()
        self.driver.switch_to.window(self.driver.window_handles[0])

        time.sleep(3)

        loc_access = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[1]')
        loc_access.click()

        time.sleep(1.5)

        notification = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[2]')
        notification.click()
        time.sleep(10)


    def like(self):
        try:
            like_button = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/div/main/div/div[1]/div/div[2]/button[3]')
        except Exception:
            like_button = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[2]/button[3]')
        like_button.click()
        time.sleep(1)


    def dislike(self):
        try :
            dislike_button = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/div/main/div/div[1]/div/div[2]/button[1]')
        except Exception:
            dislike_button = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[2]/button[1]')
        dislike_button.click()
        time.sleep(1)


    def cancelPopup(self):
        popup = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div[2]/button[2]')
        popup.click()
        time.sleep(1)

    def extractImage(self):
        try :
            pic = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/div/main/div/div[1]/div/div[1]/div[3]/div[1]/div/div/div/div/div')
        except Exception :
            pic = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[1]/div[3]/div[1]/div[1]/div/div[1]/div/div')
        link = pic.get_attribute('style').split('"')[1]

        if link != 'https://images-ssl.gotinder.com/0001unknown/640x640_pct_0_0_100_100_unknown.jpg' :
            urls = open('ImageURL.txt', 'a')
            urls.write('\n' + link)
            urls.close()

            image = requests.get(link)
            self.i += 1
            imagefile = open(self.path + 'Image%d.png' % self.i , 'wb+')
            print('Writing image No. %d.....' % self.i)
            imagefile.write(image.content)
            imagefile.close()
        else :
            pass

    def autoPilot(self):
        img = cv2.imread(self.path + 'Image%d.png' % self.i)
        faces = self.faceDetector.detect_faces(img)
        if len(faces) != 0:
            for face in faces:
                [x,y,w,h] = face['box']
                x,y,w,h = abs(x), abs(y), abs(w), abs(h)
                prediction = self.MLmodel.predict(tf.expand_dims(cv2.resize(img[y:y+h, x:x+w]/255.0, (160,160)), axis = 0))
                label = np.argmax(prediction)
                print('The prediction is : ', label)
                if label == int(self.preferGirl) and prediction[0,label] > 0.87:
                    img = cv2.putText(cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2), 'Girl %d%%' % int(prediction[0,label]*100), (x+2,y+h), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,255), thickness=1)
                    cv2.imshow('Current-Image', cv2.resize(img, (320,400)))
                    cv2.waitKey(800)
                    self.like()
                    break
                else:
                    img = cv2.putText(cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2), 'Boy %d%%' % int(prediction[0,label]*100), (x+2,y+h), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,255), thickness=1)
                    cv2.imshow('Current-Image', cv2.resize(img, (320,400)))
                    cv2.waitKey(800)
            self.dislike()
            cv2.destroyAllWindows()
        else :
            self.dislike()
            print('Failed to detect any human face.....')


bot = TinderBot()
bot.login()
print('Number of pictures present in the dataset : ', bot.i)
while True :
    try:
        bot.extractImage()
        time.sleep(1)
        bot.autoPilot()
    except Exception:
        try:
            bot.cancelPopup()
        except Exception:
            bot.dislike()