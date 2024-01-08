import setup_module
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import cv2
import numpy as np
import time
from deepface import DeepFace
from deepface.extendedmodels import Age
from modeling.abc import ModelWrapper


class DeepFaceModelWrapper(ModelWrapper):
    def __init__(self, idx, use_cache=False):
        self._idx = idx
        self._use_cache = use_cache

        self._cache_res = None
        self._name = "deepface-d{}".format(idx)
        self._model = [DeepFace.build_model('Age'), DeepFace.build_model('Race'), DeepFace.build_model('Gender')]
        self._df_gender_labels = ['Woman', 'Man']
        self._df_race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
        self._batchsize = 32
        self._backends = [
                'opencv', 
                'ssd', 
                'dlib', 
                'mtcnn', 
                'retinaface', 
                'mediapipe',
                'yolov8',
                'yunet',
            ]

    def _load_weight(self, weight_path):
        pass

    def load_cache_res(self, cache_res):
        self._cache_res = cache_res

    def predict(self, img_path):
    
        img_region_prep = [DeepFace.extract_faces(img_path[i], target_size = (224, 224), detector_backend = self._backends[0])[0]["face"] for i in range(self._batchsize)]
        img_np_series_final = np.array(img_region_prep)
        # print(img_np_series_final.shape)
        preds = [self._model[i].predict(img_np_series_final, batch_size=self._batchsize) for i in range(3)]
        preds_label = [[int(Age.findApparentAge(preds[0][i])), self._df_race_labels[np.argmax(preds[1][i])], self._df_gender_labels[np.argmax(preds[2][i])]] for i in range(self._batchsize)]
        # print(preds_label)
        return preds_label


    def cached_predict(self, idx):
        pass
        # out = self._cache_res.load(idx)

        # for k in range(len(out["class"])):
        #     out["class"][k] = cat_id_to_label(out["class"][k])

        # return out

    def get_name(self):
        return self._name
