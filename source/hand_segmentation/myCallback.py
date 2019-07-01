

# TODO
from myMetrics import *
from Utils import *
from tqdm import tqdm
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend
from sklearn.metrics import confusion_matrix


class CallbackmIoU(Callback):

    def __init__(self):
        super().__init__()
        self.mIoU = []
        self.epoch = []

    def on_epoch_begin(self, epoch, logs=None):
        
        if epoch % 2 == 0 and epoch != 0:

            print("Begin-Callback")

            # rgb validation png
            dir_img = 'Y:/tesisti/rossi/data/train_val_test_png/val_png/'
            images = glob.glob(dir_img + "*.png")
            images.sort()

            # validation groundtruth in grayscale (0-->background, 1-->class1,...)
            dir_seg = 'Y:/tesisti/rossi/data/segmentation_gray/gray/val/'
            segs = glob.glob(dir_seg + "*.png")
            segs.sort()

            mat = np.zeros(shape=(21, 21), dtype=np.int32)

            for k in tqdm(range(len(images))):
                scale = cv2.imread(images[k])
                seg = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)

                # scale = img / 127.5 - 1
                res = self.model.predict(np.expand_dims(scale, 0))

                labels = np.argmax(res.squeeze(), -1)
                labels = labels.astype(np.uint8)

                tmp = confusion_matrix(seg.flatten(), labels.flatten(),
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # class labels
                mat = mat + tmp

            iou = compute_and_print_IoU_per_class(confusion_matrix=mat, num_classes=21)

            self.mIoU.append(iou)
            self.epoch.append(epoch)
            print('The mIoU for epoch {} is {:7.2f}.'.format(epoch, iou))
