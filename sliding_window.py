'''
작성일 : 2020-09-11
작성자 : 이권동
코드 개요 : ./data/sorted/ 안에있는 데이터를 window, step 크기만큼 슬라이스 하는 코드
'''

from datetime import datetime
import numpy as np
import time
import os

filenames = os.listdir("data/sorted/")

# SVM에서 측정한 pothole
# 차량 2 개에 대해서 파일에 저장된 라인수가 해당 시간
ksc01 = [11981405, 3957425, 1398800, 1399067, 9108013, 9607348, 9619842, 9629801, 9898769, 9618378, 9894761, 9708715, 10501076, 10548083, 10765938, 11428009]
ksc02 = [6259522, 6418216, 6506414, 6506639, 6574225, 6626929, 6655555, 7273387, 7570598, 7624693, 7637556, 7789116, 7882065, 7882550, 7881607, 7853325, 9289500]

ksc01.sort()
ksc02.sort()

window_size_list = [50, 100, 150, 200]
step_list = [5, 10, 15, 20]

def sliding_step(window_size, step):
	p_sliding = []
	n_sliding = []
	path_p = "./data/npy/pothole_sliding_" + str(window_size) + "_" + str(step)
	path_n = "./data/npynormal_sliding_" + str(window_size) + "_" + str(step)
	for filename in filenames:
		p = []
		line_num = 0
		file_count = 0
		end_line_num = 400 // step
		flag = 0
	
		if "KSC01" in filename: p = ksc01
		elif "KSC02" in filename: p = ksc02
		else: p = []
		
		with open("./data/sorted/" + filename) as f:
			sq = []
			while True:
				try:
					line = f.readline()
					if not line: break
				except Exception as e:
					print(e)
					continue
				line_num += 1
				row = list(map(int, [(line.replace("\n", "").split(","))[3]]))
				sq.append(row[0])
				if len(sq) == window_size:
					if line_num in p: flag = 1
					if line_num % step == 0:
						if flag == 1:
							p_sliding.append(sq)
							end_line_num -= 1
						elif flag == 0:
							n_sliding.append(sq)

					if end_line_num == 0: 
						flag = 0
						end_line_num = 400 // step
					sq = sq[1:]
	npy_p = np.array(p_sliding)
	npy_n = np.array(n_sliding)
	print(npy_p.shape, npy_n.shape, window_size, step)

	np.save(path_p, npy_p)
	np.save(path_n, npy_n)

for ws in window_size_list:
	for s in step_list:
		sliding_step(ws, s)
