import sys
s = open(r"E:\I-observe\ui\dashboard.html", encoding="utf-8").read()
q = open(r"E:\I-observe\query_api.py", encoding="utf-8").read()
checks = [
    ("stage hidden on start", "st.style.display='none'" in s),
    ("waits for videoWidth", "if(!v||!v.videoWidth)" in s),
    ("360x360 capture", "const w=360,h=360" in s),
    ("brand I-observe", "I<span>-</span>observe" in s),
    ("no stray Y", "Y- observe" not in s),
    ("PAD_X 0.16", "PAD_X = 0.16" in q),
    ("PAD_Y 0.26", "PAD_Y = 0.26" in q),
    ("MAX_DETECT_WIDTH 360", "MAX_DETECT_WIDTH = 360" in q),
    ("analyse_faces shared", "def analyse_faces" in q),
]
for name, ok in checks:
    print(("OK   " if ok else "FAIL ") + name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
