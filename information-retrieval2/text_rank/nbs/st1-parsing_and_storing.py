
# coding: utf-8

# In[1]:

# Сделать координатный обратный индекс


# In[1]:

import multiprocessing
import time

def func(x):
    time.sleep(x)
    return x

p = multiprocessing.Pool(processes=8)
start = time.time()
for x in p.imap(func, [1,5,3, 10, 1, 15], chunksize=3):
    print("{} (Time elapsed: {}s)".format(x, int(time.time() - start)))
p.close()


# In[ ]:




# In[2]:

import os

files_paths = []
for root, dirs, files in os.walk("../processed"):
    path = root.split('/')

    for f in files:
        if "DS_Store" not in f:
            path_to_file = path + [f]
            files_paths.append('/'.join(path_to_file))


# In[3]:

from bs4 import BeautifulSoup


# In[4]:

import zlib


# In[19]:




# In[6]:

from collections import Counter


# In[7]:

def get_counter(bs):
    return Counter(bs.get_text())


# In[ ]:




# In[243]:

# Extract title + meta
# Extract body
def head_and_title_extractor(bs):
    title = bs.title.get_text() if bs.title else ''
    meat = bs.findAll(attrs={"name":"description"}) 
    meat = ' '.join([m['content'] for m in meat if 'content' in m])
    
    head = title + ' ' + meat
    
    body = bs.body.get_text() if bs.body else bs.get_text()
    
    return {
        "head": head,
        "body": body
    }


# In[244]:

files_paths[0]


# In[245]:

def files_paths_mapper(in_path):
    with open(in_path, 'rb') as fin:
        file_content = zlib.decompress(fin.read())
        
    bs = BeautifulSoup(file_content, "lxml")
    data = head_and_title_extractor(bs)
    
    file_text = data["head"] + "!@#$%^&*()" + data["body"]
    file_name = in_path.split('/')[-1].replace('html-', '').replace('txt-', '')
    file_text_compressed = zlib.compress(file_text.encode('utf-8'))
    
    out_path = "../parsed/%s" % file_name
    with open(out_path, "wb") as fout:
        fout.write(file_text_compressed)
    
    return out_path


# In[268]:

p1.close()
p1 = multiprocessing.Pool(processes=7)

mapper_gen = p1.imap(files_paths_mapper, files_paths[21900:], chunksize=30)


# In[269]:

import sys


# In[270]:

for i, path in enumerate(mapper_gen):
    sys.stdout.write('\r' + "%s | %s" % (i, path))


# In[ ]:




# In[232]:

get_ipython().system(u'open .')


# In[ ]:




# In[187]:

desc


# In[189]:

desc = b.findAll(attrs={"name":"description"}) 
print desc[0]['content']


# In[ ]:




# In[184]:

p1.close()
p1 = multiprocessing.Pool(processes=7)

def get_title(bs):
    if bs.title:
        return bs.title.get_text()
    else:
        return ''

def get_text(bs):
    return bs.get_text()

def get_bs(path_to_file):
    with open(path_to_file, 'rb') as fin:
        file_content = zlib.decompress(fin.read())
        
    return BeautifulSoup(file_content, "lxml")


title_gen = p1.imap(get_bs, files_paths[:200], chunksize=10)


# In[180]:

p3.close()
p3 = multiprocessing.Pool(processes=7)

def get_counter(text):
    tokens = text.split()
    
    return Counter(tokens)

counter_gen = p3.imap(get_counter, title_gen)


# In[181]:

sum(1 for _ in counter_gen)


# In[173]:

c = reduce(lambda x, y: x + y, counter_gen)


# In[ ]:

c.most_common()[:100]


# In[72]:

start = time.time()
ss = sum(v for v in bs_gen)
end = time.time()
print(ss, end - start)


# In[53]:




# In[32]:

bs_gen.close()


# In[31]:

bs_gen.next()a


# In[27]:




# In[46]:



counter_gen = p.imap(get_counter, bs_gen, chunksize=10)

# reduced = reduce(lambda x, y: x + y, counter_gen, Counter())
# p.close()


# In[47]:

bs_gen = p.imap(get_bs, files_paths[:100], chunksize=10)


# In[48]:

bs_gen.next()


# In[50]:

b = get_bs('../processed/html-0_-1013796998353508831')


# In[52]:

b.


# In[49]:

get_ipython().system(u"head '../processed/html-0_-1013796998353508831'")


# In[38]:

counter_gen.next()


# In[36]:

reduced.keys()[:100]


# In[ ]:



