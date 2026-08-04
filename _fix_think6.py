c = open(r'D:\vps\python\llama-cpp_vlm_web\static\js\app.js','rb').read()

# Replace from "if (_isMiniCPM) {" to (and including) "} else "
a = c.find(b'if (_isMiniCPM) {')
b = c.find(b'} else if (_thinkEnded) {', a)
b += len(b'} else ')

# Replace comment too
cmt = c.find(b'// \xe2\x94\x80\xe2\x94\x80 \xe4\xb8\xa4\xe8\xb7\xaf')
cmt_s = c.rfind(b'\r\n', 0, cmt) + 2
cmt_e = c.find(b'\r\n', cmt)

c = c[:cmt_s] + b'            // MiniCPM5 \xe8\xb5\xb0\xe6\xa0\x87\xe5\x87\x86 think-buffer (\xe7\xbc\x93\xe5\x86\xb2+\xe6\x8a\x98\xe5\x8f\xa0 <think> \xe6\xa0\x87\xe7\xad\xbe)' + c[cmt_e:]
# Adjust a,b after above change
shift = len(c) - len(open(r'D:\vps\python\llama-cpp_vlm_web\static\js\app.js','rb').read()) + (cmt_e-cmt_s) - len(b'            // MiniCPM5 \xe8\xb5\xb0\xe6\xa0\x87\xe5\x87\x86 think-buffer (\xe7\xbc\x93\xe5\x86\xb2+\xe6\x8a\x98\xe5\x8f\xa0 <think> \xe6\xa0\x87\xe7\xad\xbe)')
# Actually let's recalculate a,b
a = c.find(b'if (_isMiniCPM) {')
b = c.find(b'} else if (_thinkEnded) {', a)
b += len(b'} else ')

c = c[:a] + c[b:]

# Verify
t = c.decode('utf-8')
lines = t.split('\n')
for i in [778, 779, 780, 781, 782, 783, 784, 785]:
    if i <= len(lines):
        s = lines[i-1].encode('ascii','replace').decode('ascii')
        print(f'L{i}: {s[:140]}')

# Check for leftover _isMiniCPM in critical area
for i,l in enumerate(lines,1):
    if 'isMiniCPM' in l and 'curM' not in l:
        print(f'[WARN] leftover _isMiniCPM at L{i}: {l.strip()[:100]}')

with open(r'D:\vps\python\llama-cpp_vlm_web\static\js\app.js', 'wb') as f:
    f.write(c)
print('[OK] Saved')
