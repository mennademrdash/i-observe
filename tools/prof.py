import sys; sys.path.insert(0, r'E:\I-observe')
import config, time, cv2
from vision_core import YuNetDetector, MiniFASNet, OnnxEmbedder512, QdrantFaceIndex, expanded_crop
d=YuNetDetector('models/yunet.onnx'); l=MiniFASNet('models/minifasnet_v2.onnx')
e=OnnxEmbedder512('models/sface.onnx'); ix=QdrantFaceIndex(path=config.qdrant_path(),dim=e.dim,url=config.qdrant_url())
img=cv2.imread('data/people/menna/01.jpeg')
h,w=img.shape[:2]; nh=int(h*(640/w))
img=cv2.resize(img,(640,nh))
print('frame', img.shape, 'faces', len(d.detect(img)))
for _ in range(3):
    t=time.perf_counter(); f=d.detect(img); t1=(time.perf_counter()-t)*1000
    face=max(f,key=lambda x:x.confidence)
    t=time.perf_counter(); l.predict_detailed(expanded_crop(img,face.bbox)); t2=(time.perf_counter()-t)*1000
    t=time.perf_counter(); v=e.embed(e.align(img,face)); t3=(time.perf_counter()-t)*1000
    t=time.perf_counter(); r=ix.identify(v); t4=(time.perf_counter()-t)*1000
    print('YuNet %.0f | MiniFAS %.0f | embed %.0f | qdrant %.0f | TOTAL %.0f ms -> %s' % (t1,t2,t3,t4,t1+t2+t3+t4,r['identity']))
