#require retry, pandas, requests, xmltodict
import pandas
import requests
import tqdm
import json
import xmltodict
import re
from retry import retry
import pickle

def get_args():
    parser = argparse.ArgumentParser(description="Given a list of HTIDs, download and save MARC metadata for each HTID.")
    parser.add_argument("-c", "--csv",
                        help="CSV containing volumes to be classified, denoted by HTID")

args = get_args()
    
df = pandas.read_csv(args.csv)
pages = []

# adjust metadata below to suit your needs

def attempt(pages_list):
        for title in df.iterrows():
                try:
                        r = requests.get("https://catalog.hathitrust.org/api/volumes/full/htid/" + title[1]["htid"] + ".json")
                        y = json.loads(r.text)
                        record_position = list(y["records"].keys())[0]
                        z = xmltodict.parse(y["records"][record_position]["marc-xml"])["collection"]["record"]["datafield"]
                        temp = pages_list
                        for record in z:
                                if record["@tag"] == "300":
                                        pages_list.append(re.findall(r"([0-9]+)\sp", record["subfield"][0]["#text"])[0])
                                        print(pages_list[-1:])
                                if pages_list == temp:
                                        pages_list.append((title[1]["htid"], "N/A"))
                        print("Got " + title[1]["title"])
                except:
                        pages_list.append("N/A")
                        pass
        return pages_list

pages = attempt(pages)
df["pages"] = pages
df.to_csv(args.csv, index=None)

quit()
