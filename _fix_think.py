path = r'D:\vps\python\llama-cpp_vlm_web\static\js\app.js'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

old = '''          } else if (data.content) {
            // ?? ?????MiniCPM ?? vs ??????? ??
            if (_isMiniCPM) {
              // MiniCPM??? R1 ????? thinking/content?content ?????
              _thinkEnded = true;
              turnText += data.content;
              _tokenCount++;
              updateMsgTps(msgEl, _tokenCount, _tpsStart);
              const displayText = turnText.replace(/<tool_call\s+name="[^"]*">[\s\S]*<\/tool_call>/gi, '?? ...');
              bubble.innerHTML = bubbleBase + renderMarkdown(displayText);
              msgEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
            } else if (_thinkEnded) {'''

new = '''          } else if (data.content) {
            // MiniCPM5 ????? normal think-buffer ?????? <think> ??? ????+????
            if (_thinkEnded) {'''

if old in c:
    c = c.replace(old, new)
    compile(c, 'app.js', 'exec')
    print('[OK] MiniCPM5 think-buffer fix')
else:
    print('[ERR] old block not found')
    # debug: show what's around L780
    for i,line in enumerate(c.split('\n'), 1):
        if 778 <= i <= 793:
            safe = line.encode('ascii','replace').decode('ascii')
            print(f'  {i}: {safe[:120]}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('[OK] Saved')
