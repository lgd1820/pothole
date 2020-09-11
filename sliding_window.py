'''
작성일 : 2020-09-11
작성자 : 이권동
코드 개요 : ./data/sorted/ 안에있는 데이터를 window, step 크기만큼 슬라이스 하는 코드
'''
import numpy as np
import os

# 차량 5개가 시간 정리된 파일을 불러옴
filenames = os.listdir("./data/sorted/")

#window = [50, 100, 150, 200]
#step = [5, 10, 15, 20]

window = [200]
step = [20]


'''
    함수 개요 :
        ./data/sorted/ 안에있는 데이터를 window, step 크기만큼 슬라이스 하는 함수
    매개변수 :
        window : 윈도우 크기
		step : step 크기
'''
def data(window, step):
	pothole_list = []
	not_pothole_list = []
	for filename in filenames:

		# 데이터의 형태는 time,x,y,z,latitude,longitude 로 되어있음
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

# 여기서 여러개의 window와 step들로 slice
for w in window:
	for s in step:
		data(w, s)
