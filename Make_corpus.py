import bz2
import json
import os
import re
import xml.etree.ElementTree as ET

from six.moves import urllib
from time import time

def maybe_download(url, filename, expected_bytes=None):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print("getting from: {}".format(url))
        filename, _ = urllib.request.urlretrieve(url, filename)
    if expected_bytes:
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

bz2file = maybe_download(
    "https://dumps.wikimedia.org/idwiki/20180701/idwiki-latest-pages-articles.xml.bz2",
    "./Corpus/idwiki-latest-pages-articles.xml.bz2",
    482654054
)
print("The file is in:", bz2file)

def extract_bz2(filename):
    fname, ext = os.path.splitext(filename)
    if ext != ".bz2":
        raise ValueError("filename specified is not a .bz2")
    if os.path.exists(fname):
        print(fname, "alread existed")
        return fname

    with open(fname, "wb") as f, bz2.BZ2File(filename, "rb") as bf:
        for data in iter(lambda : bf.read(100*1024), b''):
            _ = f.write(data)
    return fname


xmlfile = extract_bz2(bz2file)
statinfo = os.stat(xmlfile)
print("file size: {:.3f} GB".format(statinfo.st_size / (1024*1024*1024)))

ns = {'export-0.1': 'http://www.mediawiki.org/xml/export-0.10/'}
tags_to_skip = ["siteinfo"]


def parse_wiki_xml(filename):
    skipping = ""
    in_page = False
    for event, elem in ET.iterparse(filename, events=("start", "end",)):
        if event == "start":
            for tag in tags_to_skip:
                if tag in elem.tag:
                    print("removing elem siteinfo")
                    skipping = tag
                    elem.clear()
                    break
            if in_page:
                continue
            if "page" in elem.tag:
                in_page = True
        elif event == "end":
            if skipping:
                if skipping in elem.tag:
                    elem.clear()
                    skipping = ""
            else:
                if "page" in elem.tag:
                    yield elem
                    elem.clear()
                    in_page = False


pages = parse_wiki_xml(xmlfile)

def process_text(text):
    # Remove any text not normally visible
    text = re.sub(r"<.*>", "", text)  # remove xml tags
    text = re.sub(r"&amp;", "&", text)  # decode URL encoded chars
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"<ref[^<]*</ref>", "", text)  # remove references <ref...> ... </ref>
    text = re.sub(r"<[^>]*>", "", text)  # remove xhtml tags
    text = re.sub(r"\[http:[^] ]*", "[", text)  # remove normal url, preserve visible text
    text = re.sub(r"\|thumb", "", text)  # remove images links, preserve caption
    text = re.sub(r"\|left", "", text)
    text = re.sub(r"\|right", "", text)
    text = re.sub(r"\|\d+px", "", text)
    text = re.sub(r"\[\[image:[^\[\]]*\|", "", text)
    text = re.sub(r"\[\[category:([^|\]]*)[^]]*\]\]", r"\1", text, flags=re.I)  # show categories without markup
    text = re.sub(r"\[\[[a-z\-]*:[^\]]*\]\]", "", text)  # remove links to other languages
    text = re.sub(r"\[\[[^\|\]]*\|", "[[", text)  # remove wiki url, preserve visible text
    text = re.sub(r"{{[^}]*}}", "", text)  # remove {{icons}} and {tables}
    text = re.sub(r"{[^}]*}", "", text)
    text = re.sub(r"\[", "", text)  # remove [ and ]
    text = re.sub(r"\]", "", text)
    text = re.sub(r"\(", "", text)  # remove ( and )
    text = re.sub(r"\)", "", text)
    text = re.sub(r"&[^;]*;", " ", text)  # remove URL encoded chars
    text = re.sub(r"\"", "", text)  # remove ' and "
    text = re.sub(r"'", "", text)
    text = re.sub(r"_", "", text)  # remove _
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"0", " nol ", text)
    text = re.sub(r"1", " satu ", text)
    text = re.sub(r"2", " dua ", text)
    text = re.sub(r"3", " tiga ", text)
    text = re.sub(r"4", " empat ", text)
    text = re.sub(r"5", " lima ", text)
    text = re.sub(r"6", " enam ", text)
    text = re.sub(r"7", " tujuh ", text)
    text = re.sub(r"8", " delapan ", text)
    text = re.sub(r"9", " sembilan ", text)
    text = text.lower()
    return text

def convert_to_text(pages):
    for i, page in enumerate(pages):
        if i % 2000 == 1999:
            print("Read {}k articles. Elapsed time: {:.3f}s".format(int((i+1)/1000), time() - t0), end="\r")

        title = page.find('export-0.1:title', ns).text.lower()
        if title.startswith("wikipedia:catatan commons"):
            page.clear()
            continue
        del title

        text = page.find('export-0.1:revision', ns).find('export-0.1:text', ns).text
        if not text:
            page.clear()
            continue

        text = process_text(text)

        words = text.split()
        del text
        total_words = len(words)
        total_long_words = len([w for w in words if len(w) > 3])
        if total_long_words < 15:
            page.clear()
            continue
        yield " ".join(words)


texts = convert_to_text(pages)

text_filename = os.path.splitext(xmlfile)[0]+".text"
print("start writing wikipedia texts to {}".format(text_filename))
t0 = time()

with open(text_filename, "wb") as f:
    for i, text in enumerate(texts):
        f.write((json.dumps({"text": text}) + os.linesep).encode("utf-8"))
        if i % 1000 == 999:
            print("Done writing {}k pages. Elapsed time: {:.3f}s".format(int((i+1)/1000), time() - t0), end="\r")
print("Done writing wikipedia texts in {:.3f}s".format(time() - t0))


# ==== Dari sini kebawah, untuk menggabungkan corpus hadits dengan corpus wikipedia ====
import json
out = open('./Corpus/corpus_all.txt', 'w') 
with open ('./Corpus/korpus.csv', 'r') as c :
    reader = c.read().split("\n")
    out.write(reader[0] + ' ')
#         print(reader[0])

def read_text_data(filename="./Corpus/idwiki-latest-pages-articles.text"):
    
    with open(filename, "rb") as f:
        i = 0
        for line in f:
            row = json.loads(line.decode("utf-8"))
            text = row["text"]
            yield text
#             print(text)
            out.write(text + ' ')
            i += 1
#             break
        print("total line: {}".format(i))
    out.close() 

for item in read_text_data():
    pass