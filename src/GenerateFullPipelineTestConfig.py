import os
import re

def classify_pth_files(folder):
    cat0 = []
    cat1 = []
    cat2 = []

    for f in os.listdir(folder):
        if not f.endswith(".pth"):
            continue

        name = f.lower()

        # ---- CATEGORY ----
        if "1x" in name:
            category = 0
        elif "28i" in name:
            category = 1
        elif "10_2x" in name:
            category = 2
        else:
            continue  # skip files that don't match any category

        # ---- FEAT ----
        cleanName = ''
        if "super" in name:
            feat = 24
            cleanName="super"
        elif "mega" in name:
            feat = 35
            cleanName="mega"
        elif "ultra" in name:
            feat = 64
            cleanName="ultra"
        else:
            feat = None  # or continue, depending on your requirement

        item = (f, category, feat, cleanName)

        if category == 0:
            cat0.append(item)
        elif category == 1:
            cat1.append(item)
        elif category == 2:
            cat2.append(item)

    return cat0, cat1, cat2

def generate_all_chains(cat0, cat1, cat2):
    chains = []

    for a in cat0:           # category 0 items
        for b in cat1:       # category 1 items
            for c in cat1:   # category 2 items
                chains.append([a, b, c])

    return chains
def fill_template(text, variables):
    def replacer(match):
        key = match.group(1)  # get variable name inside {}
        return str(variables.get(key, f"{{{key}}}"))  # keep {key} if missing

    # Find {variable_name} patterns
    return re.sub(r"\{(\w+)\}", replacer, text)

names = ["1X","28i_2X","28i_2X"]
cat1,cat2,cat3 = classify_pth_files("C:\\Users\\Fernando\\Documents\\Unicamp\\IA_376_GAN\\workspace\\dgm-2025.2\\projects\\super-video-decompression\\models\\pth")
allChains = generate_all_chains(cat1,cat2,cat3)
for chain in allChains:
    print(chain[0][0], chain[1][0], chain[2][0], "\n")
    
    fileName = chain[0][3]+names[chain[0][1]]+"-"+chain[1][3]+names[chain[1][1]]+"-"+chain[2][3]+names[chain[2][1]]
    print(fileName)
    vars = {'m1name': chain[0][0], 'm2name': chain[1][0], 'm3name': chain[2][0],
           'm1feat': chain[0][2], 'm2feat': chain[1][2], 'm3feat': chain[2][2],
           'experiment': fileName}
    
    with open('./test_triple_compact_template.toml', 'r') as file:
        filedata = file.read()
    filedata = fill_template(filedata, vars)
    
    with open("./test_triple_compact_ii/"+fileName+".toml", 'w') as file:
        file.write(filedata)
    #break
    


print (len(allChains))