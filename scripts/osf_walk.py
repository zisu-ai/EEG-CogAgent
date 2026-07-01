"""Recursively walk the OSF storage for a node and print a file manifest."""
import json
import sys
import urllib.request

NODE = sys.argv[1] if len(sys.argv) > 1 else "2v5md"
BASE = f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def walk(folder_url, prefix=""):
    out = []
    url = folder_url
    while url:
        d = get(url)
        for item in d.get("data", []):
            attrs = item["attributes"]
            name = attrs["name"]
            kind = attrs["kind"]
            path = prefix + "/" + name
            if kind == "folder":
                rel = item["relationships"]["files"]["links"]["related"]["href"]
                out.extend(walk(rel, path))
            else:
                size = attrs.get("size", 0)
                dl = item["links"].get("download", "")
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                out.append({"path": path, "size": size, "ext": ext, "download": dl})
        url = d.get("links", {}).get("next")
    return out


def main():
    files = walk(BASE + "/")
    total = sum(f["size"] for f in files)
    print(f"# Total files: {len(files)}")
    print(f"# Total size : {total} bytes ({total/1e6:.1f} MB)")
    # extension breakdown
    from collections import defaultdict
    by_ext = defaultdict(lambda: [0, 0])
    for f in files:
        by_ext[f["ext"]][0] += 1
        by_ext[f["ext"]][1] += f["size"]
    print("# By extension (count, MB):")
    for ext, (n, s) in sorted(by_ext.items(), key=lambda x: -x[1][1]):
        print(f"#   {ext or '(none)':<10} {n:>5}  {s/1e6:>10.1f}")
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
