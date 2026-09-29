from __future__ import annotations
import config  # noqa: F401  # sets BLAS thread caps before numpy/OpenCV load
import argparse, json, shutil
from pathlib import Path
import cv2, numpy as np
from vision_core import YOLO11FaceDetector, OnnxEmbedder512, TurboVecFaceIndex

def enroll(identity:str, images:list[str], destination='data/people'):
    import config
    detector=YOLO11FaceDetector(str(config.YOLO_FACE_MODEL))
    embedder=OnnxEmbedder512(str(config.embedder_path()))
    if embedder.dim != 512: raise RuntimeError(f'Active ArcFace model must output 512-D, got {embedder.dim}')
    root=Path(destination); existing=next((p for p in root.iterdir() if p.is_dir() and p.name.casefold()==identity.casefold()),None) if root.exists() else None
    folder=existing or (root/identity); folder.mkdir(parents=True,exist_ok=True)
    vectors=[]; accepted=[]; rejected=[]
    for i,source in enumerate(images,1):
        source=Path(source); target=folder/f'{i:02d}{source.suffix.lower()}'
        if not source.exists():
            rejected.append({'path':str(source),'reason':'missing_source'}); continue
        # Re-enrolling from the destination folder is normal; shutil.copy2 raises
        # SameFileError when source and target resolve to the same path.
        if source.resolve() != target.resolve():
            shutil.copy2(source,target)
        frame=cv2.imread(str(target)); faces=detector.detect(frame) if frame is not None else []
        if not faces:
            rejected.append({'path':str(target),'reason':'no_face'}); continue
        if len(faces) != 1:
            rejected.append({'path':str(target),'reason':f'expected_one_face_found_{len(faces)}'}); continue
        (face,)=faces
        x1,y1,x2,y2=face.bbox; area=max(0,x2-x1)*max(0,y2-y1)
        if area < 80*80:
            rejected.append({'path':str(target),'reason':'face_too_small'}); continue
        try: vector=embedder.embed(embedder.align(frame,face))
        except Exception as exc:
            rejected.append({'path':str(target),'reason':str(exc)}); continue
        vectors.append(vector); accepted.append({'path':str(target),'confidence':face.confidence})
    if not vectors: raise RuntimeError('No usable enrollment image')
    # Preserve each accepted pose/lighting view as its own template; a centroid can
    # reduce similarity to every real sample when enrollment views vary substantially.
    index=TurboVecFaceIndex(path=config.TURBOVEC_PATH, dim=embedder.dim)
    point_ids=[index.enroll(identity,vector) for vector in vectors]
    report={'identity':identity,'point_ids':point_ids,'accepted':accepted,'rejected':rejected,
            'vectors_used':len(vectors),'dimension':int(vectors[0].size)}
    (folder/'enrollment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('identity'); p.add_argument('images',nargs='+'); a=p.parse_args()
    print(json.dumps(enroll(a.identity,a.images),indent=2))
