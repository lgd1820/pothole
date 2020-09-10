'''
작성일 : 2020-09-10
작성자 : 이권동
코드 개요 : 저장된 버스, 위드라이브 데이터를 불러와 학습, 테스트 데이터셋으로 나누고 학습 후 모델을 저장하는 코드
'''
import numpy as np
import os

# 차량 5개가 시간 정리된 파일을 불러옴
filenames = os.listdir("./data/sorted/")

#window = [50, 100, 150, 200]
#step = [5, 10, 15, 20]

window = [200]
step = [20]

def data(window, step):
	pothole_list = []
	not_pothole_list = []
	for filename in filenames:
		with open("./data/sorted/" + filename, "r") as f:	
			l = []
			l_mode = []
			while True:
				line = f.readline()
				if not line: break
				data = line.replace("\n","").split(",")
				l.append(int(data[3]))
				l_mode.append(int(data[4]))
				if len(l) == window:
					if 1 in l_mode:
						pothole_list.append(l)
					else:
						not_pothole_list.append(l)
					l = l[step:]
					l_mode = l_mode[step:]
			print(len(pothole_list))	
	p_data = np.array(pothole_list)
	np_data = np.array(not_pothole_list)
	np.savez("./data/test_data/" + str(window) + "_" + str(step), p = p_data, n = np_data)
	print(window, step, p_data.shape, np_data.shape)
	del p_data, np_data

for w in window:
	for s in step:
		data(w, s)
