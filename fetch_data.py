"""Download the 38 NHANES public-use files analysed into data/ and check each MD5.

Cycles 2013-2014 (H), 2015-2016 (I), 2017-2018 (J) and August 2021-August 2023 (L). The checksums are those of the files
analysed (downloaded October 2026). NCHS occasionally revises a public-use file; a mismatch is reported, never
overwritten silently. Existing files in data/ are checked, not downloaded again.

  python3 fetch_data.py
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
YEAR = {'H': '2013', 'I': '2015', 'J': '2017', 'L': '2021'}
URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}"
MD5 = {
    "DEMO_H.xpt": "fe59a58ba1a5cc4814d7e6db4ebe820e",
    "BMX_H.xpt": "c5bc2b552c8cd33f10557fd16b747920",
    "WHQ_H.xpt": "4222edc03e7cce8fdcb8fb2979c7fa0d",
    "BPQ_H.xpt": "8b6463e652049319381be8cb056462d3",
    "DIQ_H.xpt": "e3e54401502f4d29f8649764268571e6",
    "MCQ_H.xpt": "9b2968321f390943fd7cb6b074b8a503",
    "TCHOL_H.xpt": "f9c6a765433bdad4f2a229fdc86bbb92",
    "GHB_H.xpt": "86ba74cc83143f1e9ea3643ae2a5b1da",
    "BPX_H.xpt": "11381ab93e54c38b16ebd5955c2832d4",
    "DEMO_I.xpt": "7c3bcc3d4dd0adf238b33016b9692ef0",
    "BMX_I.xpt": "d40ac66541b94ed27ec609e55211871a",
    "WHQ_I.xpt": "6c4c898f049f60ff1011aea7ee1513fa",
    "BPQ_I.xpt": "618e1b3dcb9e964aeb72b564d93c90d8",
    "DIQ_I.xpt": "428a382a815c10f2e11db97f0f054c9b",
    "MCQ_I.xpt": "aa003bf7879b46fd2a42145518ab3e77",
    "TCHOL_I.xpt": "891377b3a3e6ce0b6f38ab237e7eef3a",
    "GHB_I.xpt": "c5fd828c4ac9c25159959337dbffd9a0",
    "BPX_I.xpt": "ea61a348c6baff8aa2892a021cdf4e87",
    "SLQ_I.xpt": "fd91bd0d28bc9487a15dd0da9e49fdbb",
    "DEMO_J.xpt": "3d5e48e4255b2ec80578a0eb45480d4c",
    "BMX_J.xpt": "4c51e08c3b105005a326da3a4fc6b223",
    "WHQ_J.xpt": "7119c3e8698f4cbb1a50805c1d00607c",
    "BPQ_J.xpt": "73389571d1c6c3042b233b0277e4bcb3",
    "DIQ_J.xpt": "582c75421372266d1591162511ea73cb",
    "MCQ_J.xpt": "15eb9d220ee181b319b633064ebff760",
    "TCHOL_J.xpt": "8a66907b0f4d0173872a69de2b71a45d",
    "GHB_J.xpt": "3ed004ca802328e9de20bbc1b9091ae6",
    "BPX_J.xpt": "6785e6261dffcb7c060c87e520fd8df8",
    "SLQ_J.xpt": "10724d21310b4352c01b23121074c927",
    "DEMO_L.xpt": "14555b1f4e3c909172e900f63a2ecd03",
    "BMX_L.xpt": "ab25a36296e5b61881d42fef61c06b84",
    "WHQ_L.xpt": "cfc913ecaefa62fa6e25ddc23f7bb3e8",
    "BPQ_L.xpt": "83e8199b0333d87e2434399924379c52",
    "DIQ_L.xpt": "59760c8e3765f44dff6e27fef2c02e43",
    "MCQ_L.xpt": "8ddeb438ef55a9d73c7ab8bd882b21e9",
    "TCHOL_L.xpt": "2a7026e5df3f8abc9b063b243a4d321f",
    "GHB_L.xpt": "525acd4e3d18ea49b2341b43b5dc3b3d",
    "BPXO_L.xpt": "06b22e450f3b5f5b7867119331fe12a4",
}


def main():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    bad = []
    for name, md5 in MD5.items():
        path = os.path.join(ROOT, "data", name)
        if not os.path.exists(path):
            year = YEAR[name.rsplit("_", 1)[1][0]]
            req = urllib.request.Request(URL.format(year=year, name=name), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r:
                body = r.read()
            with open(path, "wb") as f:
                f.write(body)
        got = hashlib.md5(open(path, "rb").read()).hexdigest()
        print(("ok      " if got == md5 else "CHANGED ") + name)
        if got != md5:
            bad.append(name)
    if bad:
        sys.exit(f"{len(bad)} file(s) differ from the versions analysed: {', '.join(bad)}")


if __name__ == "__main__":
    main()
