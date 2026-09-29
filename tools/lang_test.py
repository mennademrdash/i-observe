import sys,json; sys.path.insert(0, r'E:\I-observe')
import config
from ai_models import vlm_smoke_test
import glob
shot = sorted(glob.glob(r'E:\I-observe\evidence\live_*\*.jpg'))[-1]
for lang in ['Egyptian Arabic slang','Levantine Arabic slang','English']:
    q = 'What is in this image? Answer in one short sentence.'
    r = vlm_smoke_test(shot, language=(lang if lang!='English' else None))
    print('[%s] ok=%s %.1fs' % (lang, r['ok'], r['elapsed_s']))
    print('   ' + (r.get('answer') or r.get('error')))
