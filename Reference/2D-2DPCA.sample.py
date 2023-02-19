'''
https://blog.csdn.net/w450468524/article/details/54895477
'''
import numpy as np 
from PIL import Image 

def PCA2D_2D(samples, row_top, col_top): 
	'''samples are 2d matrices - X''' 
	size = samples[0].shape  
	# m*n matrix 
	mean = np.zeros(size)  

	for s in samples:
		mean = mean + s  

	# get the mean of all samples 
	mean /= float(len(samples)) 

	# n*n matrix 
	cov_row = np.zeros((size[1], size[1])) 
	for s in samples: 
		diff = s - mean 
		cov_row = cov_row + np.dot(diff.T, diff)
	cov_row /= float(len(samples)) 

	row_eval, row_evec = np.linalg.eig(cov_row) 
	# select the top t evals 
	sorted_index = np.argsort(row_eval) 
	# using slice operation to reverse 
	X = row_evec[:, sorted_index[ : -row_top-1 : -1]]

	# m*m matrix 
	cov_col = np.zeros((size[0], size[0])) 
	for s in samples:
		diff = s - mean 
		cov_col += np.dot(diff, diff.T) 
	cov_col /= float(len(samples)) 
    col_eval, col_evec = np.linalg.eig(cov_col)
    sorted_index = np.argsort(col_eval)
    Z = col_evec[:,sorted_index[:-col_top-1 : -1]]

    return X, Z 



samples = [] 
for i in range(1, 6): 
	im = Image.open('iamge/'+str(i)+'.png') 
	im_data = np.empty( (im.size[1], im.size[0]) )
	for j in range(im.size[1])  : 
		for k in range(im.size[0]) :
			R = im.getpixel((k,j)) 
			im_data[j,k] = R/ 255.0 

	samples.append(im_data) 

X, Z  = PCA2D_2D(samples, 90, 90) 

res = np.dot(Z.T, np.dot(samples[0], X)) 

row_im = Image.new('L', (res.shape[1], res.shape[0])) 
y = res.reshape(1, res.shape[0] * res.shape[1]) 

row_im.putdata([int(t*255) for t in y[0].tolist()]) 
row_im.save('X.png') 





'''
https://zhuanlan.zhihu.com/p/138101583
'''


