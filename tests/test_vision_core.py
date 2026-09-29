import numpy as np
from vision_core import align_face, expanded_crop

def test_alignment_shape():
    image=np.zeros((160,160,3),np.uint8)
    landmarks=np.array([[50,60],[90,60],[70,80],[55,105],[85,105]],np.float32)
    assert align_face(image,landmarks).shape == (112,112,3)

def test_expanded_crop_is_clipped():
    image=np.zeros((100,100,3),np.uint8)
    crop=expanded_crop(image,np.array([0,0,30,30]),2.0)
    assert crop.size > 0 and crop.shape[0] <= 100 and crop.shape[1] <= 100
