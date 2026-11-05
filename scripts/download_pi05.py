import concurrent.futures, hashlib, json, pathlib, requests, time
root=pathlib.Path(__file__).resolve().parents[1]
prefix='checkpoints/pi05_libero/'
items=json.loads((root/'deployment/pi05-libero-files.json').read_text()); (root/'models/pi05_libero').mkdir(parents=True,exist_ok=True)
(root/'models/pi05_libero/source-files.json').write_text(json.dumps(items,indent=2))
def download(x):
    p=root/'models/pi05_libero'/x['name'][len(prefix):];p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size==int(x['size']):return
    part=p.with_name(p.name+'.part');offset=part.stat().st_size if part.exists() else 0
    for attempt in range(8):
        try:
            resp=requests.get(x['mediaLink'],headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=90);resp.raise_for_status()
            if offset and resp.status_code!=206:offset=0
            with part.open('ab' if offset else 'wb') as f:
                for chunk in resp.iter_content(1024*1024):f.write(chunk)
            assert part.stat().st_size==int(x['size'])
            part.rename(p);print('Downloaded',p,p.stat().st_size,flush=True);return
        except Exception as e:
            print('Retry',p,str(e),flush=True);offset=part.stat().st_size if part.exists() else 0;time.sleep(2)
    raise RuntimeError(p)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,items))
print('COMPLETE',flush=True)
