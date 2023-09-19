import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def dehaze(I,A,t, clip_min=0.1, clip_max=0.9):
   
    I = np.float32(I) / 255
 
    J = np.zeros_like(I)

    for c in range(3):
        J[:, :, c] = (I[:, :, c] - A[0, c]) / np.clip(t, clip_min, clip_max) + A[0, c]
    
    
    '''
    cv2.imshow("haze_free_image",J)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    return J
    
def est_atomspheric_light(img, d_p):

    h,w,c = img.shape
    
    img = np.float32(img) / 255
    
    # 先計算總共要找幾個pixels
    brightest = int(np.ceil(0.001*h*w))
    #np.ceil無條件進位
    
    #reshaped d_p變成一維的
    d_p_1D = d_p.reshape(1,-1)
    
    
    #從小排到大
    d_p_index = np.argsort(d_p_1D)
    
    
    img_reshape = img.reshape(1, h*w, 3)#reshaped img變成一維的
    img_red = img.copy()#標記最亮的點為紅色 #複製一份
   
    img_brightest = np.zeros((1, brightest, 3), dtype=np.float32)
        
    for i in range(brightest):
        x = d_p_index[0,h*w-1-i]#為第i個最亮的值的位置值
        img_red[int(x/w), int(x%w), 0] = 0
        img_red[int(x/w), int(x%w), 1] = 0
        img_red[int(x/w), int(x%w), 2] = 1
        #將那個位置變成紅色的
        img_brightest[0, i, :] = img_reshape[0, x, :]
        
    
    A = np.mean(img_brightest, axis=1)
    #print('atmospheric light:{}'.format(A))

    '''
    cv2.imshow("img_red",img_red)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    '''
    
    return A


def depth_map(I,size):
    #計算depth map

    I_hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = np.float32(I_hsv[:,:,1]) / 255
    v = np.float32(I_hsv[:,:,2]) / 255
    
    #高斯分布
    sigma = 0.041337
    sigma_map = np.random.normal(0, sigma, (I.shape[0], I.shape[1])).astype(np.float32)

    
    d_p_raw =  0.121779 + 0.959710 * v - 0.780245 * s + sigma_map
    
    d_p_raw = np.clip(d_p_raw, 0, None) #問題:不確定計算出來的depth map小於零的話是否要設為零，在paper中只看到d(x)範圍介於[0,+inf]
    
    '''
    #顯示跟論文相似的深度圖
    depth_map_show(d_p_raw)
    '''
    
    
    #使用最小值濾波器
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    d_p_min = cv2.erode(d_p_raw, kernel)
    #d_p_min = scipy.ndimage.minimum_filter(d_p, size)#這個也有一樣的結果
    
    '''
    #顯示跟論文相似的深度圖
    depth_map_show(d_p_min)
    '''
    
    
    return d_p_raw,d_p_min

#顯示跟論文相似的深度圖
def depth_map_show(img):
    
    min_value = np.min(img)
    max_value = np.max(img)

    # Normalize data to [0, 1] range
    normalized_img = (img - min_value) / (max_value - min_value)

    # Step 1: Create a custom colormap (from black to red to orange to white)
    colors = [(0, "black"), (0.5, "red"), (0.75, "orange"), (1, "white")]
    cmap_name = "custom_map"
    cm_custom = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    
    # Step 2: Display the image using the custom colormap
    plt.imshow(normalized_img, cmap=cm_custom)
    plt.colorbar()
    plt.axis('off')
    plt.title("Depth Map")
    plt.show()
    
    
    
    
def guided_filter(guide,img,radius,eps):
    refined_d_p = cv2.ximgproc.guidedFilter(guide=guide, src=img, radius=radius, eps=eps) #問題:為甚麼guide輸入影像必為uint8的，不能是0~1之間的
    
    '''
    #顯示跟論文相似的深度圖
    depth_map_show(refined_d_p)
    '''
    
   
    return refined_d_p

def cap(img_list,min_filer_size,guided_filter_r,epsilon):

    #-----------------------------------------------------
    #參數設定
    #min_filer_size = 15 #最小值濾波器大小 paper有給資訊   
    beta = 1.0 #散射系数 paper有給資訊
    #guided_filter_r = 20 #引導濾波器半徑為2*r+1 因為半徑大小必須為奇數 paper有給資訊
    #epsilon = 10**-3 #引導濾波器epsilon的值  paper有給資訊
    #------------------------------------------------------

    dehazing_img = []

    for I in img_list:
 
        #計算depth map
        d_p_raw, d_p_min = depth_map(I,min_filer_size)
        
        #對經過最小濾波器的depth map進行引導濾波器濾波
        d_p_min_guide =  guided_filter(I,d_p_min,guided_filter_r,epsilon)
        
        #計算atmospheric light
        A = est_atomspheric_light(I, d_p_min_guide)
        
        #轉成transmission map
        transmission_map= np.exp(-beta * d_p_min_guide)   

        #除霧
        J = dehaze(I, A, transmission_map)
        #cv2.imwrite("haze-free.png", J*255)

        dehazing_img.append(J)

    return dehazing_img
if __name__ == '__main__':

    img_fog = [] 

    #輸入影像路徑    
    img_path1 = (r'image\haze\6.png')
    #讀取影像
    img_fog.append(cv2.imread(img_path1))

    #輸入影像路徑    
    img_path2 = (r'image\haze\7.png')
    #讀取影像
    img_fog.append(cv2.imread(img_path2))



    
    J = cap(img_fog,15,20,10**-3)
   