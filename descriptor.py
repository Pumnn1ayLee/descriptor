import os
import dlib
import glob
import numpy
import cv2

# 1.人脸关键点检测器路径
predictor_path = "E://Download/shape_predictor_68_face_landmarks.dat.dat"
# 2.人脸识别模型路径
face_rec_model_path = "E://Download/dlib_face_recognition_resnet_model_v1.dat.dat"
# 3.待识别的人脸路径
img_path = "E://Download/predictor"

# 1.创建正脸检测器detector
detector = dlib.get_frontal_face_detector()
# 2.创建人脸关键点检测器sp
sp = dlib.shape_predictor(predictor_path)
# 3.创建人脸识别器facerec
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 候选人名单
candidate = ['Zhourunfa','TheShy','ZhangziYi']

descriptors = numpy.load('descriptor.npy') # 加载候选人描述子列表
for f in glob.glob(os.path.join(img_path,"*.jpg")):
    img = cv2.imread(f)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    dets = detector(img,1) # 待识别人脸检测
    for k,d in enumerate(dets):
        dist = []
        shape = sp(img,d)
        face_descriptor = facerec.compute_face_descriptor(img,shape)
        d_test = numpy.array(face_descriptor) # 待识别人脸的描述子
        # 计算描述子差值
        for i in descriptors:    # 逐个候选人遍历，i为每个候选人的描述子
            dist_ = numpy.linalg.norm(i-d_test) #计算差值
            dist.append(dist_)
        c_d = dict(zip(candidate,dist)) # 候选人和距离组成一个dict
        cd_sorted = sorted(c_d.items(),key=lambda d:d[1]) # 升序排列
        if cd_sorted[0][1]<0.5:
            print('\n The person is: %s'%(cd_sorted[0][0]))
            cv2.putText(img, "{0}".format(cd_sorted[0][0], i), (d.left(), d.top()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
        else:
            print('\n The person is unknown')
            cv2.putText(img, "unknown", (d.left(), d.top()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('a', img)
    c = cv2.waitKey(0)
    if c == ord('q'):  # key值太高，会导致视频帧数很低
        continue
