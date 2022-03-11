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
    datelist.append(date)
    insertions.append(numins)
    deletions.append(-numdel)
insertions = numpy.array(insertions)
deletions = numpy.array(deletions)
total = insertions + deletions
total = total[::-1].cumsum()[::-1]

from matplotlib import pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(datelist, insertions, color='green', ds='steps', label='Insertions')
ax1.plot(datelist, deletions, color='red', ds='steps', label='Deletions')
ax2.plot(datelist, total, color='purple', ds='steps', label='Cumulative')
ax1.axhline()
ax1.grid()
ax1.legend()
ax2.grid()
ax2.tick_params(labelrotation=45 )
fig.tight_layout()
fig.savefig('gitstats.pdf')
