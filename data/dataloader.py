import torchaudio
from torch.utils.data import Dataset
import cv2
import PIL
import random
import numpy as np
from packaging import version
from PIL import Image
import os
import torch
import pandas as pd
import pickle
import json


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------
imagenet_templates_small = [
    "a photo of {}"
]


class VGGSound(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            logger,
            size=512,
            interpolation='bicubic',
    ):
        video_lst = "video/"
        audio_lst = "audio/"

        self.video = args.data_dir + video_lst
        self.audio = args.data_dir + audio_lst
        print("数据集视频文件路径：", self.video)
        print("数据集音频文件路径：", self.audio)
        self.vggsound = "data/VGGSound/vggsound2.csv"
        self.video_path = list()
        self.audio_path = list()
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = args.placeholder_token
        self.data_set = args.data_set
        self.df = pd.read_csv(self.vggsound)
        self.input_length = args.input_length
        if self.data_set == 'train':
            self.center_crop = args.center_crop
            self.filter_frames = args.filter_frames
            self.filter_unmatch_videos = args.filter_unmatch_videos
            self.filter_low_quality_imgs = args.filter_low_quality_imgs
        else:
            self.center_crop = False
            self.filter_frames = False
            self.filter_unmatch_videos = False
            self.filter_low_quality_imgs = False

        with open('filter_config/optimal_frames_data.json', 'r') as file:
            self.frames = json.load(file)

        videos = set([file_path[:-4] for file_path in os.listdir(self.video)])
        audios = set([file_path[:-4] for file_path in os.listdir(self.audio)])
        samples = videos & audios
        
        if self.data_set == 'train' and self.filter_unmatch_videos:
            with open("filter_config/mismatched_video_pairs.pkl", "rb") as file:
                filtered_out = pickle.load(file)
                self.df = self.df[~self.df['ytid'].isin(filtered_out)]

        if self.data_set == 'train' and self.filter_low_quality_imgs:
            with open("filter_config/poor_quality_video_list.pkl", "rb") as file:
                filtered_out = pickle.load(file)
                self.df = self.df[~self.df['ytid'].isin(filtered_out)]

        self.label = list()
        self.prepare_dataset(samples)

        self.num_samples = len(self.video_path)

        self._length = self.num_samples

        logger.info(f"{args.data_set}, num samples: {self.num_samples}")

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.templates = imagenet_templates_small
        # self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def prepare_dataset(self, samples):
        df = self.df[self.df["set"] == self.data_set]
        for vid in list(samples):
            video = vid[:11]
            df_vid = df[df.ytid == video]
            if df_vid.empty:
                continue
            label = df_vid["class"].unique()[0]
            self.video_path.append(os.path.join(self.video, vid + ".mp4"))
            self.audio_path.append(os.path.join(self.audio, vid + ".wav"))
            self.label.append(label)
            
    def cleanup_files(self):
        """清理数据集中不匹配或低质量的音视频文件"""
        videos_to_remove = set(os.listdir(self.video)) - set([os.path.basename(path) for path in self.video_path])
        audios_to_remove = set(os.listdir(self.audio)) - set([os.path.basename(path) for path in self.audio_path])

        for vid in videos_to_remove:
            vid_path = os.path.join(self.video, vid)
            if os.path.exists(vid_path):
                os.remove(vid_path)

        for aud in audios_to_remove:
            aud_path = os.path.join(self.audio, aud)
            if os.path.exists(aud_path):
                os.remove(aud_path)
                
    def sample_frame(self, video_path, rand_sec):
        # Open the speech_video file
        video = cv2.VideoCapture(video_path)
        
        # 检查是否成功打开了视频
        if not video.isOpened():
            print(f"Failed to open video file: {video_path}")
            return None
        
        # Get the total number of frames in the speech_video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"Video has no frames: {video_path}")
            video.release()
            return None

        low, high = (total_frames // 10) * rand_sec, (total_frames // 10) * (rand_sec+self.input_length)
        ytid = video_path.split('/')[-1][:11]
        candidates = []
        
        if any(ytid == key[:11] for key in self.frames):
            try:
                actual_key = next(key for key in self.frames if ytid == key[:11])
                candidates = self.frames[actual_key][0]
                candidates = [frame for frame in candidates if low <= frame <= high]
            except:
                candidates = []
        
        # Set the speech_video to the chosen frame
        frame_num = random.choice(candidates) if candidates else random.randint(low, high)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = video.read()
            
        # Release the speech_video file
        video.release()
        # Return the frame
        return frame

    def img_proc(self, vid, rand_sec):
        image = self.sample_frame(vid, rand_sec)
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((512, 320), resample=self.interpolation)

        # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)

    def aud_proc_beats(self, aud, rand_sec=0):
        wav, sr = torchaudio.load(aud)
        wav = torch.tile(wav, (1, 10))
        wav = wav[:, :sr*10]
        start = rand_sec * sr
        end = (rand_sec+self.input_length) * sr
        wav = wav[:, start:end]
        return wav[0]

    def txt_proc(self):
        text = random.choice(self.templates).format(self.placeholder_token)
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, i):
        example = {}
        example["input_ids"] = self.txt_proc()
        ytid = self.audio_path[i % self.num_samples].split('/')[-1][:11]
        example["ytid"] = ytid
        example['label'] = self.label[i % self.num_samples]
        example['full_name'] = self.audio_path[i % self.num_samples].split('/')[-1][:-4]
        aud = self.audio_path[i % self.num_samples]
        if self.input_length == 10:
            rand_sec = 0
        else:
            rand_sec = np.random.randint(0, 10 - self.input_length)

        vid = self.video_path[i % self.num_samples]
        example["pixel_values"] = self.img_proc(vid, rand_sec)
        example["audio_values"] = self.aud_proc_beats(aud, rand_sec)
        return example


