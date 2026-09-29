from __future__ import annotations
import config  # noqa: F401  # sets BLAS thread caps before numpy/OpenCV load
import argparse, json
from pathlib import Path
import cv2
from vision_core import YOLO11FaceDetector, MiniFASNet, OnnxEmbedder512, TurboVecFaceIndex, expanded_crop

def run(source, output, sample_every=3):
    import config
    detector=YOLO11FaceDetector(str(config.YOLO_FACE_MODEL)); live_model=MiniFASNet(str(config.LIVENESS_PATH))
    embedder=OnnxEmbedder512(str(config.embedder_path()))
    if embedder.dim != 512: raise RuntimeError(f'Active ArcFace model must output 512-D, got {embedder.dim}')
    index=TurboVecFaceIndex(path=config.TURBOVEC_PATH, dim=embedder.dim)
    cap=cv2.VideoCapture(source); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    w,h=int(cap.get(3)),int(cap.get(4)); Path(output).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    frame_no=0; events=[]; last=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame_no%sample_every==0:
            last=[]
            for face in detector.detect(frame):
                is_live,live_score=live_model.predict(expanded_crop(frame,face.bbox))
                identity={'identity':'spoof_rejected','score':0.0}
                if is_live:
                    try: identity=index.identify(embedder.embed(embedder.align(frame,face)))
                    except Exception as exc: identity={'identity':'alignment_error','score':0.0}
                x1,y1,x2,y2=map(int,face.bbox); item={'frame':frame_no,'time':frame_no/fps,'bbox':[x1,y1,x2,y2],'identity':identity['identity'],'similarity':identity['score'],'live':is_live,'liveness_score':live_score,'landmarks':face.landmarks.tolist()}
                events.append(item); last.append(item)
        for item in last:
            x1,y1,x2,y2=item['bbox']
            label=item['identity']
            if not item['live']:
                color=(0,0,255)
            elif label in ('unknown','no_face','alignment_error','spoof_rejected'):
                color=(0,180,255)
            else:
                color=(0,220,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            for lx,ly in item.get('landmarks',[]): cv2.circle(frame,(int(lx),int(ly)),2,color,-1)
            text=f"{label} sim={item['similarity']:.2f} live={item['liveness_score']:.2f}"
            cv2.putText(frame,text,(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        writer.write(frame); frame_no+=1
    cap.release(); writer.release()
    identified=[e for e in events if e['identity'] not in ('unknown','no_face','alignment_error','spoof_rejected')]
    summary={'source':str(source),'output':str(Path(output).resolve()),'frames':frame_no,'detections':len(events),
             'identified_faces':len(identified),
             'identities':sorted({e['identity'] for e in identified}),
             'live':sum(e['live'] for e in events),'spoof_rejected':sum(not e['live'] for e in events),
             'max_similarity':max((e['similarity'] for e in events),default=0),'events':events}
    Path(output).with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--output',default='outputs/annotated.mp4'); a=p.parse_args()
    result=run(a.source,a.output); print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
