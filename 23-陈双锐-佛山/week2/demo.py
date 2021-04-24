# coding:utf8
# 基于numpy实现余弦相似度计算，实现batchnormalization层前向计算。调试一下demo任务，熟悉pytorch训练过程
import numpy as np;


def cosine_sim(x, y):
	z = np.dot(x, y)
	denom = np.linalg.norm(x) * np.linalg.norm(y)
	cos = z / denom
	return cos


x = np.array([1,1])
y = np.array([1,0])
sim = cosine_sim(x, y)
print(sim)