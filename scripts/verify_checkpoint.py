import base64,hashlib,json,pathlib,google_crc32c
r=pathlib.Path(__file__).resolve().parents[1];m=r/'models/pi05_libero';items=json.loads((m/'source-files.json').read_text());results=[]
for x in items:
 p=m/x['name'].split('checkpoints/pi05_libero/')[1];h=hashlib.md5();s=hashlib.sha256();c=google_crc32c.Checksum()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);s.update(b);c.update(b)
 actual=base64.b64encode(h.digest()).decode();crc=base64.b64encode(c.digest()).decode();assert crc==x['crc32c'],str(p)
 if 'md5Hash' in x:assert actual==x['md5Hash'],str(p)
 results.append({'file':str(p.relative_to(r)),'size':p.stat().st_size,'generation':x['generation'],'md5_base64':actual,'crc32c_base64':crc,'sha256':s.hexdigest()});print('verified',p.name,flush=True)
(r/'models/pi05_libero/verified-hashes.json').write_text(json.dumps(results,indent=2))
