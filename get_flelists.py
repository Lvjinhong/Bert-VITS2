# ch_name
import os
import click



def process(ch_name, ch_language="ZH"):
    with open(out_file, "a", encoding="Utf-8") as wf:
        path = os.path.join(base_path, ch_name)
        files = os.listdir(path)
        for f in files:
            if f.endswith(".lab"):
                with open(os.path.join(path, f), "r", encoding="utf-8") as perFile:
                    line = perFile.readline()
                    result = f"{Base}/dataset/{ch_name}/{f.split('.')[0]}.wav|{ch_name}|{ch_language}|{line}"
                    wf.write(f"{result}\n")


# @click.command()
# @click.option('--base-path', default="raw",type=str, help='None')
# @click.option('--out-file', default="filelists/genshin_out.txt", help='The person to greet.')
def main():
    names = [
        it for it in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, it))
    ]
    print(names)
    for ch_name in names:
        process(ch_name, out_file)


if __name__ == "__main__":
    # 根据 raw 和 lab 文件 去得到转录文件列表
    Base = ""
    base_path = os.path.join(Base,"raw")
    out_file = os.path.join(Base,"filelists/genshin/genshin.list")

    main()
