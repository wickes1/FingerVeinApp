from ctypes import Structure, c_int

FTR_OPTIONS_RESERVED_V1 = 0x00800000


class FTRSCAN_IMAGE_SIZE(Structure):
    _fields_ = [("nWidth", c_int), ("nHeight", c_int), ("nImageSize", c_int)]


def load_device():
    # load api
    from ctypes import WinDLL, c_void_p, byref, c_uint8, c_bool
    import platform
    import os

    os_platform = platform.system()
    if os_platform == "Linux":
        scanAPI = CDLL("./libScanAPI.so")
    elif os_platform == "Darwin":
        scanAPI = CDLL("./libScanAPI.dylib")
    elif os_platform == "Windows":
        folder = os.path.dirname(os.path.abspath(__file__))
        # print(folder+"/api/ftrScanAPI.dll")
        scanAPI = WinDLL(folder + "/api/ftrScanAPI.dll")
    scanAPI.ftrScanOpenDevice.restype = c_void_p
    # open device
    hDevice = c_void_p(scanAPI.ftrScanOpenDevice())
    if hDevice is None:
        print("Failed to open device!")
    # get image size

    img_size = FTRSCAN_IMAGE_SIZE()
    if not scanAPI.ftrScanGetImageSize(hDevice, byref(img_size)):
        print("Failed to get image size!")
        scanAPI.ftrScanCloseDevice(hDevice)
    # set options
    scanAPI.ftrScanSetOptions(hDevice, FTR_OPTIONS_RESERVED_V1, FTR_OPTIONS_RESERVED_V1)
    # print("width:%d, height:%d, imageSize:%d" %
    #       (img_size.nWidth, img_size.nHeight, img_size.nImageSize))
    # allocate buffer for getting frame
    buftype = c_uint8 * img_size.nImageSize
    pBuffer = buftype()
    scanAPI.ftrScanGetFrame.argtype = [c_void_p, c_uint8, c_void_p]
    scanAPI.ftrScanGetFrame.restype = c_bool

    scanAPI.ftrScanGetImage2.argtype = [c_void_p, c_uint8, c_void_p]
    scanAPI.ftrScanGetImage2.restype = c_bool

    return scanAPI, hDevice, pBuffer, img_size


def displayWindowOnTop():
    import cv2
    import numpy

    WindowName = "Main View"
    view_window = cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)

    # These two lines will force the window to be on top with focus.
    cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    img = numpy.zeros((400, 400, 3), numpy.uint8)
    cv2.imshow(WindowName, img)
    cv2.destroyWindow(WindowName)


def make_folder():
    import os

    folderList = ["evaluation_result", "pickle", "record", "record_second"]
    for name in folderList:
        try:
            os.mkdir(name)
        except:
            pass


if __name__ == "__main__":
    pass