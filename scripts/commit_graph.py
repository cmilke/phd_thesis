import subprocess
import re
import numpy
import datetime


gitlog = subprocess.run(['git', 'log', '--stat'],stdout=subprocess.PIPE).stdout.decode('utf-8')

datelist = []
insertions = []
deletions = []
for commit in gitlog.split('commit'):
    commitparse = re.search('^([^\n]*).*Date:\s*([^\n]*).*\n\s*[0-9]+ file[s]? changed,\s*(([0-9]+) insertion[s]?\(\+\))?,?\s*(([0-9]+) deletion[s]?\(-\))?', commit, re.DOTALL)
    if commitparse is None: continue
    infolist = commitparse.groups()
    commitID = infolist[0].strip()
    date = datetime.datetime.strptime(infolist[1],'%c %z')
    numins = 0 if infolist[3] is None else int(infolist[3])
    numdel = 0 if infolist[5] is None else int(infolist[5])
    if commitID == '5fa901dbe1d4a47c83853ad91f95d296d0662fcd': continue
    if commitID == 'b5d606c59faf07736f83d7af84c134775d15c50e' : continue
    if commitID == 'b04717c92bbc9b300ae13a1b7c36bf7995c80865' : continue
    datelist.append(date)
    insertions.append(numins)
    deletions.append(-numdel)
insertions = numpy.array(insertions)
deletions = numpy.array(deletions)
total = insertions + deletions
total = total[::-1].cumsum()[::-1]

from matplotlib import pyplot as plt
numbins = 30
fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':(2,1)})
#ax1.plot(datelist, insertions, color='green', ds='steps', label='Insertions')
#ax1.plot(datelist, deletions, color='red', ds='steps', label='Deletions')
#ax2.plot(datelist, total, color='purple', ds='steps', label='Cumulative')
inserted, bins = ax1.hist(datelist, weights=insertions, bins=numbins, color='green', histtype='stepfilled', label='Insertions')[:2]
deleted = ax1.hist(datelist, weights=deletions, bins=numbins, color='red', histtype='stepfilled', label='Deletions')[0]
total = (inserted + deleted).cumsum()
ax2.hist(bins[:-1], weights=total, bins=bins, color='purple', histtype='stepfilled', label='Cumulative')
ax1.axhline()
ax1.grid()
ax1.legend(loc='upper left')
ax2.grid()
ax2.tick_params(labelrotation=45 )
fig.tight_layout()
fig.savefig('gitstats.pdf')
