#!/usr/bin/env python


import json, re, sys

def getSplitMap(d):
    parts = []
    for s in d:
        p = re.split('(\.|\?|\!)', s)
        # Removing extra part
        if not p[-1]:
            p = p[:-1]
        # Marking all dividers in sentence   
        for pp in p:
            parts.append([pp, 1 if re.match('(\.|\?|\!)', pp) else 0])
        # Last divider in sentence will be marked as -1 
        # Other dividers will be marked as 1
        if parts[-1][1]:
            parts[-1][1] = -1
            
    # Very last divider isn't real divider, so we won't count it
    if parts[-1][1]:
        parts[-1][1] = 0
    
    # for pp in parts:
    #     print pp[0], pp[1]

    # print [pp[1] for pp in parts]
    return parts
        
def compareSplit(gld, spl):
    g_parts = getSplitMap(gld['Sentences'])
    s_parts = getSplitMap(spl['Sentences'])
    
    #tp, tn, fp, fn
    res = [0, 0, 0, 0]
    failed_res = [0, 0, sum([abs(x[1]) for x in g_parts if x[1] < 0]), sum([abs(x[1]) for x in g_parts if x[1] > 0])]
    
    # Check splitting correctness
    if len(g_parts) != len(s_parts):
        return failed_res
    for a, b in zip(g_parts, s_parts):
        if abs(a[1] - b[1]) == 1:
            return failed_res
    
    # Count splitting
    for a, b in zip(g_parts, s_parts):
        if not a[1] and not b[1]:
            continue
        if a[1] + b[1] == -2:
            res[0] += 1
        elif a[1] + b[1] == 2:
            res[1] += 1
        elif a[1] > 0 and b[1] < 0:
            res[2] += 1
        elif a[1] < 0 and b[1] > 0:
            res[3] += 1
        else:
            return failed_res
    return res

fg = open(sys.argv[1])
gs = [json.loads(s) for s in fg.readlines()]
fs = open(sys.argv[2])
ss = [json.loads(s) for s in fs.readlines()]

cc = [0, 0, 0, 0]
c1 = 0
for gld, spl in zip(gs, ss):
    r = compareSplit(gld, spl)
    
    cc = [a + b for a, b in zip(cc,r)]
    
print 'Accuracy:',float(sum(cc[:2]))/sum(cc)

