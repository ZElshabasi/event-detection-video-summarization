# Event Detection and Video Summarization

An assignment project exploring video event detection and summary generation with SmolVLM2 and CLIP. The main deliverable is a Jupyter notebook that guides the workflow from video frames to event descriptions, temporal retrieval, evaluation, and a shortened video summary.

## Project workflow

The notebook contains these stages:

1. Sample frames from a video and use **SmolVLM2** with prompt constraints to describe salient events.
2. Parse event descriptions and compare them with human annotations using sentence-embedding similarity. It also includes Cohen's kappa calculations for matched annotator events.
3. Extract frame embeddings with **CLIP** and retrieve candidate moments using fixed-window top-k scoring and an adaptive-threshold method. Although one notebook heading mentions Moment-DETR, the included implementation uses direct CLIP frame similarities and does not import Moment-DETR.
4. Compare retrieved moments with annotated time intervals using intersection over union (IoU), mean IoU, and Recall@0.5.
5. Merge candidate intervals and use MoviePy to create shortened summary videos.

The notebook preserves the assignment's terminology and exploratory code. Its saved outputs, progress widgets, and model download logs have been cleared so the repository contains the notebook source without large generated output.

## Repository contents

```text
.
├── Event_Detection_Video_summarization_pipeline.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

The notebook was authored for Google Colab and expects GPU acceleration for practical use with the 2.2B parameter vision-language model. It can also be run locally if the machine has enough memory and a compatible PyTorch installation.

1. Use a Python version supported by your selected PyTorch and Transformers builds. The notebook metadata records Python 3.13.9 from its original runtime; Google Colab is the closest match to that environment.
2. Install PyTorch for your operating system and CPU/CUDA setup using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
3. Install the remaining packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open the notebook in Jupyter or upload it to Colab. The notebook's first cells also contain the original inline package-install commands.
5. Provide the required video files and annotation files in a location available to the notebook, then update the path variables in the notebook to match that location.

The model checkpoints are downloaded from Hugging Face on first use. This requires network access and disk space. The SmolVLM2 model requires substantial memory; a compatible GPU is recommended. The final video export also requires an FFmpeg-capable MoviePy installation.

## Inputs and outputs

This repository does not include video files, human annotation files, model weights, cached embeddings, or generated summary videos. The assignment's test videos and annotations were shared at the [Surfdrive folder](https://surfdrive.surf.nl/s/DAS4bz2ZKj3JY2Z). Download them only if you can access the share and follow the dataset and video owners' terms. A public download link does not by itself grant permission to rehost the files.

### Prepare videos and annotations

The notebook expects four SumMe clips, named `video_20.mp4` through `video_23.mp4`, and two annotation files for each clip. Keep the files outside the repository (the `.gitignore` excludes common video and data folders and file types).

For Google Colab:

1. Download the four video files and eight annotation text files from the Surfdrive folder to your computer. The annotation filenames include a video number and an annotator identifier.
2. Upload the video files to the Colab runtime so they are available as `/content/video_20.mp4`, `/content/video_21.mp4`, `/content/video_22.mp4`, and `/content/video_23.mp4`.
3. For each video, make local copies of its two annotation files and give the copies these notebook-friendly names:

   ```text
   18-SumMe-video_20-annotator1.txt
   18-SumMe-video_20-annotator2.txt
   18-SumMe-video_21-annotator1.txt
   18-SumMe-video_21-annotator2.txt
   18-SumMe-video_22-annotator1.txt
   18-SumMe-video_22-annotator2.txt
   18-SumMe-video_23-annotator1.txt
   18-SumMe-video_23-annotator2.txt
   ```

   Keep each video's two annotators paired together. `annotator1` and `annotator2` are local aliases used by the notebook; assign them consistently for that video. Upload the renamed copies to `/content/` in Colab. You can instead keep the original filenames and edit `ANN1_PATH`, `ANN2_PATH`, and the annotation paths in the full-pipeline cell to match.
4. For a single-video run, change `video_path` and the annotation paths in the notebook to the video and annotation pair you want to evaluate. The default `video_path` is `/content/video_21.mp4`; the IoU example uses video 20. The final batch cell processes all four videos and expects the names above.

For local Jupyter, place videos and annotation files in local folders of your choice, then update the notebook's Colab-style `/content/...` paths to those locations. Do not commit the files, even if you rename local annotation copies for convenience.

The notebook writes generated summary videos to the current working directory. Treat model descriptions and automatically retrieved intervals as candidates for review; they may be incomplete or inaccurate and depend on prompts, frame sampling, model versions, and hardware.

## Dependencies

`requirements.txt` lists the notebook's non-PyTorch packages. PyTorch is installed separately because its correct build depends on the user's hardware and operating system. The notebook uses `moviepy.editor` and `VideoFileClip.subclip`, so MoviePy is constrained to the compatible 1.x API.

## Data, licensing, and attribution

- The uploaded archive included a report PDF and eight annotation text files. They are not included in this repository. The text-file names contain annotator/student identifiers, and the report may contain course or contributor information; get permission before sharing those materials publicly.
- The Surfdrive link is provided as the source for the assignment's test videos and annotations. Its current access settings and redistribution terms have not been verified here. Check the dataset provider's terms and the rights for each original video before using or redistributing them. Do not assume the dataset is covered by this repository's code terms.
- The notebook loads `HuggingFaceTB/SmolVLM2-2.2B-Instruct` and `openai/clip-vit-base-patch32` from Hugging Face. Review each model's card and license before use or redistribution. SmolVLM2 checkpoints are published under Apache-2.0 according to the [model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct).
- The code is based on collaborative assignment work. No project license is included because the archive does not establish permission from every contributor to relicense or redistribute the work. Obtain agreement from all rights holders before adding an open-source license or publishing the repository.

## Responsible interpretation

This is an educational and exploratory project, not a validated video understanding system. Similarity thresholds and temporal retrieval scores are heuristics. Event descriptions may hallucinate details; retrieved moments may not correspond to the intended event; and generated summaries can omit important context. Review outputs before relying on or sharing them.
