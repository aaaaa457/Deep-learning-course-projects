## 项目的执行步骤

### 1. 环境准备
安装库(torchaudio torchvision transformers>=4.25.1 diffusers accelerate ftfy Pillow datasets opencv-python)：执行“library_installation.ipynb”

https://huggingface.co/camenduru/beats/blob/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt 下载BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt到models/beats文件夹

### 2. 数据集准备
先从 https://huggingface.co/datasets/Loie/VGGSound 下载VGGSound数据集（含vggsound.csv），做相关处理：执行“data_prepare.ipynb”

生成过滤配置文件（filter_config/optimal_frames_data.json，filter_config/poor_quality_video_list.pkl，filter_config/mismatched_video_pairs.pkl）：执行“filter_files_generate.ipynb”

### 3. 训练
训练出output/VGGSound_features.bin文件保存音频特征的嵌入：执行“train_VGGSound.ipynb”

### 4. 生成图片
用到稳定扩散模型compvis/stable-diffusion-v1-4（默认路径pretrained_model_name_or_path从Hugging Face官网加载，可在ModelScope魔搭社区下载模型到本地运行）：执行“generate_images.ipynb”