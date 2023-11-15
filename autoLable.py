from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from config import config

"""
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='./Model/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)

rec_result = inference_pipeline(audio_in='ge_1570_2.wav')
print(rec_result)
# {'text': '欢迎大家来体验达摩院推出的语音识别模型'}
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--in_dir", default=config.mypreprocess_config.autoLable_config.in_dir
)
parser.add_argument(
    "-n",
    "--character_name",
    default=config.mypreprocess_config.autoLable_config.character_name,
    help="人物名，对应到dataset的地址映射",
)
parser.add_argument(
    "-o", "--out_file", default=config.mypreprocess_config.autoLable_config.out_file
)
args = parser.parse_args()
in_dir = args.in_dir
character_name = args.character_name
out_file = args.out_file
local_dir_model = "./Model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

dir_path = os.path.join(*out_file.split("/")[:-1])


if not os.path.exists(dir_path):
    print(f"{dir_path} 不存在, 已创建...")
    os.makedirs(dir_path)


complete_list = []
filelist = list(os.listdir(in_dir))
# print(filelist)
if os.path.exists(out_file):
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            pt = line.strip().split("|")[0]
            complete_list.append(pt)
# print(complete_list)

inference_pipeline = pipeline(task=Tasks.auto_speech_recognition, model=local_dir_model)

from tqdm import tqdm

for file in tqdm(filelist):
    if file[-3:] != "wav":
        tqdm.write(f"{file} not supported, ignoring...\n")
        continue
    tqdm.write(f"transcribing {in_dir +'/'+ file}...\n")
    if not character_name:
        character_name = file.rstrip(".wav").split("_")[0]
    # 先生成raw_list这样的文件，所以路径先填raw，后面会clean raw 得到dataset中的数据
    savepth = os.path.join(config.dataset_path, "raw", character_name, file)

    if savepth in complete_list:
        tqdm.write(f"{file} is already done, skip!")
        continue

    rec_result = inference_pipeline(audio_in=os.path.join(in_dir, file))

    if "text" not in rec_result:
        tqdm.write("Text is not recognized，ignoring...\n")
        continue

    annos_text = rec_result["text"]
    annos_text = "[ZH]" + annos_text.replace("\n", "") + "[ZH]"
    annos_text = annos_text + "\n"
    line1 = savepth + "|" + character_name + "|" + annos_text
    line2 = savepth + "|" + character_name + "|ZH|" + rec_result["text"] + "\n"
    # with open("./long_character_anno.txt", 'a', encoding='utf-8') as f:
    #     f.write(line1)
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(line2)
    tqdm.write(rec_result["text"])
print("Done!\n")
