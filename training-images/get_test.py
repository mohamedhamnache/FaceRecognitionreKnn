import os
for dirname, dirnames, _ in os.walk('./'):
    for l in dirnames:
         filepath = os.path.join(dirname, l)
         cpt = 0
         for f in os.listdir(filepath):
             cpt=cpt+1
             if(cpt>2):
                break
         if (cpt==2):
            os.system('rm -r ' +filepath)