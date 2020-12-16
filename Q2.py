import cv2
import numpy as np

def corner(img,k,nx,ny):
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    object_pt = []
    img_pt = []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corner = cv2.findChessboardCorners(img, (nx,ny))


    if ret == True:
        object_pt.append(objp)
        img_pt.append(corner)
        cv2.drawChessboardCorners(img,(nx,ny), corner, ret)
        img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)),interpolation=cv2.INTER_LINEAR)
        cv2.imshow('corner'+str(k), img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()
    # calibrate the camera
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_pt, img_pt, img_size,None,None)
    mtx = np.reshape(mtx,(3,3))
    print(np.squeeze(rvecs,None))
    print(np.squeeze(tvecs,None))

    rvecs,_ = cv2.Rodrigues(np.squeeze(rvecs,None))
    tvecs = np.reshape(tvecs,(3,1))
    extrinsic = np.concatenate([rvecs, tvecs],axis=1)
    return mtx,dist,extrinsic



if __name__=='__main__':

    for i in range(1,16):
        img = cv2.imread('Q2_Image/'+str(i)+'.bmp')
        nx = 11
        ny = 8
        mtx,dist,ext= corner(img,i,nx,ny)
        print('intrinsic:',mtx)
        print('extrinsic',ext)
        print("distorsion",dist)

    cv2.destroyAllWindows()