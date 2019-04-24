import cv2
import time
def step1_显示图片():
    img_origin = cv2.imread('cxk.jpg')
    # 显示图片,窗口名origin1
    cv2.imshow('origin1', img_origin)

    # waitKey 等待时间ms,按键后关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def frame_resolve(img_origin):
    #灰度图
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    #高斯模糊图
    #使用的(5,5)参数就表示高斯核的尺寸，这个核尺寸越大图像越模糊。但是记住尺寸得是奇数！这是为了保证中心位置是一个像素而不是四个像素。
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    """
    在代码中我们用cv2.adaptiveThreshold()来实现这种自适应二值化方法。
    其中参数255表示我们二值化后图像的最大值，cv2.ADAPTIVE_THRESH_GAUSSIAN_C表示我们采用的自适应方法，
    cv2.THRESH_BINARY表示我们是将大于阈值的像素点的值变成最大值，
    反之这里如果使用cv2.THRESH_BINARY_INV表示我们是将大于阈值的像素点的值变成0，
    倒数第二个参数5表示我们用多大尺寸的区块来计算阈值，倒数第一个参数2表示计算周边像素点均值时待减去的常数C。

    """
    img_threshold1 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    #二值化后再次模糊
    img_threshold1_blurred = cv2.GaussianBlur(img_threshold1, (5, 5), 0)
    #再次二值,其中200表示将图片中像素值为200以上的点都变成255，255就是白色。这样我们就能得到一个边线更宽的二值化效果
    _, img_threshold2 = cv2.threshold(img_threshold1_blurred, 200, 255, cv2.THRESH_BINARY)
    #下面让我们去掉图片中一些细小的噪点，这种效果可以通过图像的开运算来实现：
    """
    图像膨胀就是腐蚀的反向操作，把图像中的区块变大一圈，把瘦子变成胖子。
    因此当我们对一个图像先腐蚀再膨胀的时候，一些小的区块就会由于腐蚀而消失，
    再膨胀回来的时候大块区域的边线的宽度没有发生变化，这样就起到了消除小的噪点的效果。
    图像先腐蚀再膨胀的操作就叫做开运算。

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_opening = cv2.bitwise_not(cv2.morphologyEx(cv2.bitwise_not(img_threshold2), cv2.MORPH_OPEN, kernel))
    #再次模糊
    img_opening_blurred = cv2.GaussianBlur(img_opening, (3, 3), 0)
    return img_opening_blurred

def show_video(video_name):
    cap = cv2.VideoCapture(video_name)

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

        img_threshold1 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,
                                               2)

        img_threshold1_blurred = cv2.GaussianBlur(img_threshold1, (5, 5), 0)

        _, img_threshold2 = cv2.threshold(img_threshold1_blurred, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_opening = cv2.bitwise_not(cv2.morphologyEx(cv2.bitwise_not(img_threshold2), cv2.MORPH_OPEN, kernel))

        img_opening_blurred = cv2.GaussianBlur(img_opening, (3, 3), 0)
        cv2.imshow('img_opening_blurred', img_opening_blurred)

        #40ms 一秒25桢
        if cv2.waitKey(40) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
if __name__ == '__main__':
    # step1_显示图片()
    # step2_图片转灰度图()
    # show_video(r'C:\Users\xudabiao\Downloads\Compressed\ywoyeesbusq.mkv')
    # 捕获摄像头
    cap = cv2.VideoCapture(0)
    # 定义编解码器，创建VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            print(type(frame))
            rF = frame_resolve(frame)
            print(type(rF))
            out.write(frame_resolve(frame))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1000//20) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()







