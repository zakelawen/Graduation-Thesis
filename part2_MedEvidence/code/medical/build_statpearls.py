import argparse
import json
import os
import re
import tarfile
import xml.etree.ElementTree as ET
MAX_WORDS = 200
MIN_WORDS = 60

def text_of(node):
    return re.sub('\\s+', ' ', ''.join(node.itertext())).strip()

def paragraphs(sec, heading=''):
    title = sec.find('title')
    heading = text_of(title) if title is not None else heading
    for child in sec:
        if child.tag == 'sec':
            yield from paragraphs(child, heading)
        elif child.tag in ('p', 'list'):
            t = text_of(child)
            if t:
                yield (heading, t)

def split_long(text):
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    sents = re.split('(?<=[.!?])\\s+', text)
    (out, cur) = ([], [])
    for s in sents:
        if cur and len(cur) + len(s.split()) > MAX_WORDS:
            out.append(' '.join(cur))
            cur = []
        cur.extend(s.split())
    if cur:
        out.append(' '.join(cur))
    return out

def chunks_for_article(root):
    title = ''
    for path in ('.//book-part-meta//title', './/title-group/title', './/title'):
        node = root.find(path)
        if node is not None and text_of(node):
            title = text_of(node)
            break
    body = root.find('.//body')
    if body is None:
        return (title, [])
    (out, buf, buf_head) = ([], [], None)
    for (heading, para) in paragraphs(body):
        for piece in split_long(para):
            if buf and (heading != buf_head or len(' '.join(buf).split()) + len(piece.split()) > MAX_WORDS):
                out.append((buf_head, ' '.join(buf)))
                buf = []
            buf_head = heading
            buf.append(piece)
            if len(' '.join(buf).split()) >= MIN_WORDS and len(' '.join(buf).split()) >= MAX_WORDS * 0.75:
                out.append((buf_head, ' '.join(buf)))
                buf = []
    if buf:
        out.append((buf_head, ' '.join(buf)))
    return (title, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    n_articles = n_chunks = 0
    with tarfile.open(args.tar) as tar, open(os.path.join(args.out_dir, 'statpearls.jsonl'), 'w', encoding='utf-8') as out:
        for member in tar:
            if not member.name.endswith('.nxml'):
                continue
            data = tar.extractfile(member).read()
            try:
                root = ET.fromstring(data)
            except ET.ParseError:
                continue
            (title, chunks) = chunks_for_article(root)
            art = os.path.basename(member.name)[:-5]
            for (i, (heading, content)) in enumerate(chunks):
                content = f'{heading}. {content}' if heading else content
                out.write(json.dumps({'id': f'{art}_{i}', 'title': title, 'content': content}, ensure_ascii=False) + '\n')
            n_articles += 1
            n_chunks += len(chunks)
    print(f'articles {n_articles}, snippets {n_chunks}')
if __name__ == '__main__':
    main()
