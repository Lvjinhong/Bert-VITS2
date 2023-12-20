

from transformers import ClapModel, ClapProcessor
#单独下载下来，解决clap wraaper中模型在本地找不到的问题
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused",cache_dir="./emotional")
ClapModel.from_pretrained("laion/clap-htsat-fused",cache_dir="./emotional")
print(processor)
