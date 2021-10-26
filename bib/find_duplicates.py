import glob
import re

def matchenclosing(line):
    bracecount = 0
    quotecount = 0
    for char in line:
        if char == '"': quotecount += 1
        if char == '{': bracecount += 1
        if char == '}': bracecount -= 1
    braceclosed = bracecount == 0
    quoteclosed = quotecount % 2 == 0
    return braceclosed and quoteclosed

bibfilelist = glob.glob('*.bib')

rawtitlelist = []
for bibfile in bibfilelist:
    with open(bibfile) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'title' in line:
                j = 0
                while not matchenclosing(line):
                    j += 1
                    line = line.strip()+' '+lines[i+j].strip()
                rawtitlelist.append(line)

titlelist = []
regexpattern = '\s*title\s*=\s*"?{?(.*}")?(.*")?(.*})?(.*),'
for rawtitle in rawtitlelist:
    patternmatch = re.search(regexpattern, rawtitle)
    groups = patternmatch.groups()
    titlestring = ''
    if groups[0] is not None: titlestring += groups[0][:-2]
    if groups[1] is not None: titlestring += groups[1][:-1]
    if groups[2] is not None: titlestring += groups[2][:-1]
    titlestring += groups[3]
    titlelist.append(titlestring)

duplicatelist = []
uniquetitles = set()
for title in titlelist:
    if title in uniquetitles:
        duplicatelist.append(title)
    else:
        uniquetitles.add(title)
print('\n'.join(duplicatelist))
#print('\n'.join(titlelist))
