# Implementing Audio-to-Image Generation Based on Diffusion Model

## Project Overview
This project is a deep learning-based system for generating images from audio, aiming to produce images related to the given audio signals.

## Environment Setup
Install libraries: library_installation.ipynb

Download from [BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt](https://huggingface.co/camenduru/beats/blob/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt) to the models/beats folder.

## Dataset Preparation
Download the VGGSound dataset (including vggsound.csv) from [VGGSound](https://huggingface.co/datasets/Loie/VGGSound)，process it: data_prepare.ipynb
 
Generate filter configuration files: filter_files_generate.ipynb

## Training
Train to save audio feature embeddings in output/VGGSound_features.bin: train_VGGSound.ipynb

## Images Generation
Utilize the stable diffusion model [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)  for image generation:
generate_images.ipynb

The generated images are shown in output/imgs folder.
<table>
  <tr>
    <td>
      <img src="output/imgs/1L_QllvdK74_000030.png" alt="Image 1" style="width:100%;">
      <p>car passing by</p>
    </td>
    <td>
      <img src="output/imgs/1MhjSKooAZo_000300.png" alt="Image 2" style="width:100%;">
      <p>ocean burbling</p>
    </td>
    <td>
      <img src="output/imgs/2a6AytwygrI_000100.png" alt="Image 3" style="width:100%;">
      <p>playing electronic organ</p>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
      <img src="output/imgs/2hhaxOZmJsY_000694.png" alt="Image 1" style="width:100%;">
      <p>airplane</p>
    </td>
    <td>
      <img src="output/imgs/2yb5ojhk8rk_000157.png" alt="Image 2" style="width:100%;">
      <p>barn swallow calling</p>
    </td>
    <td>
      <img src="output/imgs/3Qzk1nQ3a7Q_000070.png" alt="Image 3" style="width:100%;">
      <p>waterfall burbling</p>
    </td>
  </tr>
</table>

## Reference Code
[audio-to-image](https://github.com/rishavroy97/audio-to-image/tree/main)

[audio-diffusion](https://github.com/teticio/audio-diffusion)
