"""Public export: reproduce deduplicated R-squared counts only.
The original significance-count block was excluded because its float-only
p-value parser drops strings such as <0.001 and multi-SE annotations.
See VALIDATION.md. No regressions or source data are rerun here.
"""
import csv, collections
from pathlib import Path
P=Path(__file__).resolve().parents[2] / 'phase2/output/table_C_h2.csv'
rows=list(csv.DictReader(open(P)))
def f(x):
    try: return float(x)
    except: return None
# cell = country|episode|unit|window ; prefer 总回报 dep_type
cells=collections.OrderedDict()
for r in rows:
    key=(r['country'],r['episode'],r['unit'],r['window'])
    pref = 0 if r['dep_type'].startswith('总回报') else 1
    if key not in cells or pref < cells[key][0]:
        cells[key]=(pref,r)
sel=[(k,v[1]) for k,v in cells.items()]
print('total cells:',len(sel))
elig=[]
for k,r in sel:
    nn=f(r['ngdp_n']); rn=f(r['rgdp_n']); nr=f(r['ngdp_r2']); rr=f(r['rgdp_r2'])
    if None in (nn,rn,nr,rr): continue
    if nn>=8 and rn>=8: elig.append((k,r,nr,rr))
print('eligible cells (n>=8 both):',len(elig))
w=sum(1 for _,_,nr,rr in elig if nr>rr); t=sum(1 for _,_,nr,rr in elig if nr==rr)
print('ngdp>rgdp: %d  tie: %d  lose: %d  share=%.3f'%(w,t,len(elig)-w-t,w/len(elig)))
print()
print('--- per eligible cell ---')
for k,r,nr,rr in elig:
    mark='N>' if nr>rr else ('=' if nr==rr else 'R>')
    print('%-4s %-4s %-11s %-46s %-42s n=%s/%s  ngdpR2=%.2f rgdpR2=%.2f | b_n=%s p_n=%s | b_r=%s p_r=%s'%(
        mark,k[0],k[1],k[3][:44],k[2][:40],r['ngdp_n'],r['rgdp_n'],nr,rr,r['ngdp_beta'],r['ngdp_p'],r['rgdp_beta'],r['rgdp_p']))
print()
print('--- by window group ---')
def grp(wname):
    for g in ['bust 前','bust 中','恢复后 ②','恢复后 ③','触底后全程']:
        if wname.startswith(g): return g
    return wname
gd=collections.defaultdict(lambda:[0,0])
for k,r,nr,rr in elig:
    g=grp(k[3]); gd[g][1]+=1
    if nr>rr: gd[g][0]+=1
for g,(a,b) in gd.items(): print('%-12s %d/%d'%(g,a,b))
print()
print('--- by country ---')
cd=collections.defaultdict(lambda:[0,0])
for k,r,nr,rr in elig:
    cd[k[0]][1]+=1
    if nr>rr: cd[k[0]][0]+=1
for c,(a,b) in cd.items(): print('%s %d/%d'%(c,a,b))
print()
print("Significance-count aggregation omitted: p-value parsing and standard-error policy require reconciliation; see VALIDATION.md")
