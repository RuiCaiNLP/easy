#coding=utf-8
import time
from googletrans import Translator
name = ['translate.google.com', 'translate.google.fr', 'translate.google.cn']
name_idxx = 0
translator = Translator(service_urls=[name[name_idxx%3]])
name_idxx+=1
source_file = open("./processed/train_pro", 'r')
dest_file = open("translate_fr", 'w')
index = 0
num =0
for line in source_file.readlines():
    if len(line) < 2:
        index = 0
        continue
    if index == 0:
        line = line.strip()
        text = translator.translate(line,src='en',dest='fr').text
        dest_file.write(text.encode('utf-8'))
        dest_file.write("\n")
        num+=1
        if num % 500 ==0:
            print(num)
            if num % 10 ==0:
                time.sleep(1)
            translator = Translator(service_urls=[name[name_idxx%3]])
            name_idxx += 1
        index = 1
